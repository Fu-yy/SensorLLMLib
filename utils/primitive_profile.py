import os
import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ============================================================
# Basic utilities
# ============================================================

def pad_to_multiple(x: torch.Tensor, multiple: int, pad_value: float = 0.0):
    """
    x: [B, L, C]
    return:
        x_pad: [B, L_pad, C]
        L_orig: original length
    """
    if x.dim() != 3:
        raise ValueError(f"x should be [B, L, C], got {tuple(x.shape)}")

    B, L, C = x.shape
    L_pad = ((L + multiple - 1) // multiple) * multiple

    if L_pad == L:
        return x, L

    pad_len = L_pad - L
    pad = x.new_full((B, pad_len, C), pad_value)
    return torch.cat([x, pad], dim=1), L


def pad_mask_to_len(mask: Optional[torch.Tensor], B: int, L_orig: int, L_pad: int, device):
    """
    padding_mask: [B, L], True/1 means valid.
    return: [B, L_pad] bool
    """
    if mask is None:
        out = torch.zeros((B, L_pad), device=device, dtype=torch.bool)
        out[:, :L_orig] = True
        return out

    mask = mask.to(device)
    if mask.dtype != torch.bool:
        mask = mask > 0

    if mask.shape[1] == L_pad:
        return mask

    if mask.shape[1] > L_pad:
        return mask[:, :L_pad]

    pad_len = L_pad - mask.shape[1]
    pad = torch.zeros((B, pad_len), device=device, dtype=torch.bool)
    return torch.cat([mask, pad], dim=1)


def get_label_names_from_cfg(ds_cfg: Optional[Dict[str, Any]], num_class: Optional[int] = None):
    """
    Supports:
        ds_cfg["label_names"] = [...]
        ds_cfg["id2label"] = {0: "...", 1: "..."}
    """
    if not isinstance(ds_cfg, dict):
        return None

    if "label_names" in ds_cfg and ds_cfg["label_names"] is not None:
        return [str(x) for x in ds_cfg["label_names"]]

    if "id2label" in ds_cfg and ds_cfg["id2label"] is not None:
        id2label_raw = ds_cfg["id2label"]

        if not isinstance(id2label_raw, dict):
            raise TypeError(f"ds_cfg['id2label'] should be dict, got {type(id2label_raw)}")

        id2label = {}
        for k, v in id2label_raw.items():
            id2label[int(k)] = str(v)

        if num_class is None:
            num_class = max(id2label.keys()) + 1

        missing = [i for i in range(int(num_class)) if i not in id2label]
        if len(missing) > 0:
            raise ValueError(
                f"id2label missing ids: {missing}. "
                f"Available keys: {sorted(id2label.keys())}"
            )

        return [id2label[i] for i in range(int(num_class))]

    return None


def get_channel_names_from_cfg(ds_cfg: Optional[Dict[str, Any]], channel_num: int):
    if isinstance(ds_cfg, dict) and "channel_names" in ds_cfg and ds_cfg["channel_names"] is not None:
        names = [str(x) for x in ds_cfg["channel_names"]]
        if len(names) == channel_num:
            return names

    return [f"channel_{i}" for i in range(channel_num)]


# ============================================================
# Segment-level statistics
# ============================================================

def safe_float(x, default: float = 0.0):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


@torch.no_grad()
def estimate_periodicity(seg: torch.Tensor) -> float:
    """
    seg: [T, C]

    Use autocorrelation of motion magnitude.
    Return roughly [0, 1].
    Larger means more periodic/repetitive local motion.
    """
    if seg.dim() != 2 or seg.shape[0] < 4:
        return 0.0

    mag = torch.sqrt((seg ** 2).mean(dim=-1) + 1e-8)  # [T]
    mag = mag - mag.mean()

    denom = torch.sum(mag ** 2).clamp_min(1e-8)

    max_lag = max(2, seg.shape[0] // 2)
    vals = []

    for lag in range(1, max_lag):
        v = torch.sum(mag[:-lag] * mag[lag:]) / denom
        vals.append(v)

    if len(vals) == 0:
        return 0.0

    return safe_float(torch.stack(vals).max().clamp(min=0.0, max=1.0).item())


@torch.no_grad()
def spectral_entropy(seg: torch.Tensor) -> float:
    """
    seg: [T, C]

    Return normalized spectral entropy in [0, 1].
    Lower means more concentrated frequency pattern.
    Higher means more irregular/broadband.
    """
    if seg.dim() != 2 or seg.shape[0] < 4:
        return 0.0

    x = seg - seg.mean(dim=0, keepdim=True)
    fft = torch.fft.rfft(x, dim=0).abs() ** 2  # [F, C]

    if fft.shape[0] <= 1:
        return 0.0

    # remove DC component
    power = fft[1:, :].mean(dim=1)  # [F-1]
    power = power.clamp_min(1e-12)
    prob = power / power.sum().clamp_min(1e-12)

    ent = -(prob * torch.log(prob)).sum()
    ent = ent / math.log(prob.numel() + 1e-12)

    return safe_float(ent.clamp(0.0, 1.0).item())


@torch.no_grad()
def dominant_frequency_index(seg: torch.Tensor) -> int:
    """
    Return dominant non-DC frequency index.
    This is not Hz, only FFT bin index within the local segment.
    """
    if seg.dim() != 2 or seg.shape[0] < 4:
        return 0

    x = seg - seg.mean(dim=0, keepdim=True)
    fft = torch.fft.rfft(x, dim=0).abs() ** 2  # [F, C]

    if fft.shape[0] <= 1:
        return 0

    power = fft[1:, :].mean(dim=1)  # remove DC
    idx = int(torch.argmax(power).item()) + 1

    return idx


@torch.no_grad()
def zero_crossing_rate(seg: torch.Tensor) -> float:
    """
    Simple sign-change rate over time.
    """
    if seg.dim() != 2 or seg.shape[0] < 2:
        return 0.0

    x = seg - seg.mean(dim=0, keepdim=True)
    signs = torch.sign(x)
    changes = (signs[1:] * signs[:-1]) < 0

    return safe_float(changes.float().mean().item())


@torch.no_grad()
def compute_segment_stats(seg: torch.Tensor) -> Dict[str, Any]:
    """
    seg: [T, C]

    Return segment-level physical statistics.
    """
    if seg.dim() != 2:
        raise ValueError(f"seg should be [T, C], got {tuple(seg.shape)}")

    T, C = seg.shape
    abs_seg = seg.abs()

    energy = safe_float((seg ** 2).mean().item())
    rms = safe_float(torch.sqrt((seg ** 2).mean() + 1e-8).item())
    mean_abs = safe_float(abs_seg.mean().item())
    std = safe_float(seg.std(unbiased=False).item())
    peak_abs = safe_float(abs_seg.max().item())

    if T > 1:
        delta = seg[1:] - seg[:-1]
        temporal_variation = safe_float(delta.abs().mean().item())
        delta_energy = safe_float((delta ** 2).mean().item())
    else:
        temporal_variation = 0.0
        delta_energy = 0.0

    if T > 2:
        jerk = seg[2:] - 2 * seg[1:-1] + seg[:-2]
        jerk_abs = safe_float(jerk.abs().mean().item())
    else:
        jerk_abs = 0.0

    channel_energy = (seg ** 2).mean(dim=0)  # [C]
    dominant_channel = int(torch.argmax(channel_energy).item())

    channel_energy_sum = channel_energy.sum().clamp_min(1e-8)
    channel_energy_ratio = channel_energy / channel_energy_sum

    stats = {
        "energy": energy,
        "rms": rms,
        "mean_abs": mean_abs,
        "std": std,
        "peak_abs": peak_abs,
        "temporal_variation": temporal_variation,
        "delta_energy": delta_energy,
        "jerk_abs": jerk_abs,
        "periodicity": estimate_periodicity(seg),
        "spectral_entropy": spectral_entropy(seg),
        "dominant_frequency_index": dominant_frequency_index(seg),
        "zero_crossing_rate": zero_crossing_rate(seg),
        "dominant_channel": dominant_channel,
        "dominant_channel_energy_ratio": safe_float(channel_energy_ratio[dominant_channel].item()),
        "channel_energy_ratio": [safe_float(x) for x in channel_energy_ratio.detach().cpu().tolist()],
    }

    return stats


# ============================================================
# Online accumulator
# ============================================================

class PrimitiveAccumulator:
    def __init__(self, num_codes: int, num_channels: int, keep_examples_per_code: int = 5):
        self.num_codes = int(num_codes)
        self.num_channels = int(num_channels)
        self.keep_examples_per_code = int(keep_examples_per_code)

        self.code_count = torch.zeros(self.num_codes, dtype=torch.long)
        self.code_label_count = [Counter() for _ in range(self.num_codes)]
        self.dominant_channel_count = [Counter() for _ in range(self.num_codes)]
        self.transition_prev_count = [Counter() for _ in range(self.num_codes)]
        self.transition_next_count = [Counter() for _ in range(self.num_codes)]

        self.scalar_sum = defaultdict(float)
        self.scalar_sq_sum = defaultdict(float)
        self.scalar_count = defaultdict(int)

        self.channel_ratio_sum = torch.zeros(self.num_codes, self.num_channels, dtype=torch.float64)
        self.channel_ratio_count = torch.zeros(self.num_codes, dtype=torch.long)

        self.examples = [[] for _ in range(self.num_codes)]

        self.total_tokens = 0
        self.total_sequences = 0

    def add_scalar(self, cid: int, key: str, value: float):
        value = safe_float(value)
        self.scalar_sum[(cid, key)] += value
        self.scalar_sq_sum[(cid, key)] += value * value
        self.scalar_count[(cid, key)] += 1

    def scalar_mean_std(self, cid: int, key: str) -> Tuple[Optional[float], Optional[float]]:
        n = self.scalar_count.get((cid, key), 0)
        if n <= 0:
            return None, None

        mean = self.scalar_sum[(cid, key)] / n
        var = self.scalar_sq_sum[(cid, key)] / n - mean * mean
        std = math.sqrt(max(var, 0.0))

        return safe_float(mean), safe_float(std)

    def add_occurrence(
        self,
        cid: int,
        label_id: int,
        token_pos: int,
        sequence_index: int,
        segment_stats: Dict[str, Any],
        prev_id: Optional[int] = None,
        next_id: Optional[int] = None,
        label_name: Optional[str] = None,
    ):
        if cid < 0 or cid >= self.num_codes:
            return

        self.code_count[cid] += 1
        self.code_label_count[cid][int(label_id)] += 1
        self.total_tokens += 1

        dom_ch = int(segment_stats.get("dominant_channel", 0))
        self.dominant_channel_count[cid][dom_ch] += 1

        if prev_id is not None and 0 <= int(prev_id) < self.num_codes:
            self.transition_prev_count[cid][int(prev_id)] += 1

        if next_id is not None and 0 <= int(next_id) < self.num_codes:
            self.transition_next_count[cid][int(next_id)] += 1

        scalar_keys = [
            "energy",
            "rms",
            "mean_abs",
            "std",
            "peak_abs",
            "temporal_variation",
            "delta_energy",
            "jerk_abs",
            "periodicity",
            "spectral_entropy",
            "dominant_frequency_index",
            "zero_crossing_rate",
            "dominant_channel_energy_ratio",
        ]

        for key in scalar_keys:
            if key in segment_stats:
                self.add_scalar(cid, key, segment_stats[key])

        ch_ratio = segment_stats.get("channel_energy_ratio", None)

        if ch_ratio is not None and len(ch_ratio) == self.num_channels:
            self.channel_ratio_sum[cid] += torch.tensor(ch_ratio, dtype=torch.float64)
            self.channel_ratio_count[cid] += 1

        # keep representative examples:
        # prioritize high-energy and high-purity-like examples by energy
        if self.keep_examples_per_code > 0:
            example = {
                "sequence_index": int(sequence_index),
                "token_pos": int(token_pos),
                "label_id": int(label_id),
                "label_name": label_name if label_name is not None else f"class_{label_id}",
                "energy": safe_float(segment_stats.get("energy", 0.0)),
                "rms": safe_float(segment_stats.get("rms", 0.0)),
                "periodicity": safe_float(segment_stats.get("periodicity", 0.0)),
                "temporal_variation": safe_float(segment_stats.get("temporal_variation", 0.0)),
                "dominant_channel": int(segment_stats.get("dominant_channel", 0)),
            }

            self.examples[cid].append(example)
            self.examples[cid] = sorted(
                self.examples[cid],
                key=lambda x: x["energy"],
                reverse=True,
            )[:self.keep_examples_per_code]


# ============================================================
# Semantic description helpers
# ============================================================

def percentile(values: List[float], q: float, default: float = 0.0):
    values = [safe_float(x) for x in values if x is not None and not math.isnan(float(x))]
    if len(values) == 0:
        return default

    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    idx = max(0, min(idx, len(values) - 1))

    return values[idx]


def level_by_quantile(value: float, low_thr: float, high_thr: float, names: Tuple[str, str, str]):
    value = safe_float(value)

    if value < low_thr:
        return names[0]
    if value < high_thr:
        return names[1]
    return names[2]


def support_level(count: int):
    count = int(count)
    if count <= 0:
        return "unused"
    if count < 30:
        return "low-support"
    if count < 100:
        return "medium-support"
    return "high-support"


def build_distribution(counter: Counter, total: int, label_names: Optional[List[str]] = None, topk: Optional[int] = None):
    out = []

    if total <= 0:
        return out

    items = counter.most_common(topk)

    for label_id, count in items:
        label_id = int(label_id)

        if label_names is not None and 0 <= label_id < len(label_names):
            label_name = str(label_names[label_id])
        else:
            label_name = f"class_{label_id}"

        out.append({
            "label_id": label_id,
            "label_name": label_name,
            "count": int(count),
            "ratio": safe_float(count / total),
        })

    return out


def build_transition_distribution(counter: Counter, total: int, topk: int = 5):
    if total <= 0:
        return []

    out = []

    for code_id, count in counter.most_common(topk):
        out.append({
            "code_id": int(code_id),
            "count": int(count),
            "ratio": safe_float(count / total),
        })

    return out


def make_description(
    count: int,
    stats: Dict[str, Any],
    label_distribution: List[Dict[str, Any]],
    quantiles: Dict[str, Tuple[float, float]],
):
    if count <= 0:
        return "an unused motion primitive in the current training split"

    support = stats.get("support_level", support_level(count))

    energy = safe_float(stats.get("energy_mean", 0.0))
    temporal = safe_float(stats.get("temporal_variation_mean", 0.0))
    periodicity = safe_float(stats.get("periodicity_mean", 0.0))
    spec_ent = safe_float(stats.get("spectral_entropy_mean", 0.0))
    channel_name = str(stats.get("dominant_channel_name", "an unspecified channel"))
    channel_ratio = safe_float(stats.get("dominant_channel_ratio", 0.0))
    purity = safe_float(stats.get("label_purity", 0.0))

    e_low, e_high = quantiles.get("energy", (0.0, 1.0))
    t_low, t_high = quantiles.get("temporal_variation", (0.0, 1.0))
    p_low, p_high = quantiles.get("periodicity", (0.0, 1.0))
    se_low, se_high = quantiles.get("spectral_entropy", (0.0, 1.0))

    intensity = level_by_quantile(
        energy,
        e_low,
        e_high,
        ("low-intensity", "moderate-intensity", "high-intensity"),
    )

    dynamics = level_by_quantile(
        temporal,
        t_low,
        t_high,
        ("stable", "smoothly varying", "rapidly changing"),
    )

    periodic = level_by_quantile(
        periodicity,
        p_low,
        p_high,
        ("non-periodic", "weakly periodic", "strongly periodic"),
    )

    entropy_level = level_by_quantile(
        spec_ent,
        se_low,
        se_high,
        ("spectrally concentrated", "moderately broadband", "spectrally complex"),
    )

    if len(label_distribution) == 0:
        label_part = "without a reliable activity association"
    else:
        top = label_distribution[0]
        label_name = top["label_name"]
        ratio = safe_float(top["ratio"])

        if purity < 0.35:
            label_part = f"shared across multiple activities, with weak association to {label_name} ({ratio:.2f})"
        elif purity < 0.55:
            label_part = f"moderately associated with {label_name} ({ratio:.2f})"
        else:
            label_part = f"strongly associated with {label_name} ({ratio:.2f})"

    return (
        f"a {support}, {intensity}, {dynamics}, {periodic} motion primitive, "
        f"{entropy_level}, dominated by {channel_name} "
        f"(channel dominance={channel_ratio:.2f}), {label_part}"
    )


def make_case_study_summary(
    primitive_id: int,
    count: int,
    stats: Dict[str, Any],
    label_distribution: List[Dict[str, Any]],
):
    if count <= 0:
        return {
            "title": f"Primitive {primitive_id}: unused primitive",
            "summary": "This primitive is not activated in the current training split.",
            "interpretation": "It should not be used for semantic interpretation unless activated in another split or dataset.",
        }

    top_label = label_distribution[0]["label_name"] if len(label_distribution) > 0 else "unknown activity"
    purity = safe_float(stats.get("label_purity", 0.0))
    energy = safe_float(stats.get("energy_mean", 0.0))
    periodicity = safe_float(stats.get("periodicity_mean", 0.0))
    channel_name = str(stats.get("dominant_channel_name", "unknown channel"))

    if purity >= 0.55:
        role = "discriminative"
    elif purity >= 0.35:
        role = "moderately discriminative"
    else:
        role = "shared"

    summary = (
        f"Primitive {primitive_id} appears {count} times and is mainly associated with "
        f"{top_label} (purity={purity:.2f}). It shows average energy={energy:.4f}, "
        f"periodicity={periodicity:.4f}, and is dominated by {channel_name}."
    )

    interpretation = (
        f"This primitive can be treated as a {role} local motion pattern. "
        f"In case studies, it can be used to explain why a sequence is biased toward "
        f"{top_label}, especially when this primitive repeatedly appears in the primitive sequence."
    )

    return {
        "title": f"Primitive {primitive_id}: {role} cue for {top_label}",
        "summary": summary,
        "interpretation": interpretation,
    }


# ============================================================
# Main profile builder
# ============================================================

@torch.no_grad()
def build_strong_primitive_profile(
    vq_model,
    dataloader,
    save_path: str,
    device,
    num_codes: int = 512,
    label_names: Optional[List[str]] = None,
    channel_names: Optional[List[str]] = None,
    max_batches: Optional[int] = None,
    keep_examples_per_code: int = 5,
    min_valid_ratio_per_token: float = 0.5,
    include_meta: bool = True,
):
    """
    Generate a strong primitive profile JSON for paper-level interpretability.

    Required model method:
        vq_model.get_token_ids(x) -> [B, P]

    Dataloader batch formats supported:
        (batch_x, label, padding_mask)
        {"x": ..., "label": ..., "padding_mask": ...}

    Output JSON:
        {
          "_meta": {...},
          "0": {
            "id": 0,
            "count": ...,
            "frequency": ...,
            "label_distribution": [...],
            "stats": {...},
            "description": "...",
            "case_study": {...},
            "examples": [...]
          },
          ...
        }
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if max_batches is not None:
        max_batches = int(max_batches)
        if max_batches <= 0:
            max_batches = None

    vq_model = vq_model.to(device)
    vq_model.eval()

    # infer stride
    if hasattr(vq_model, "stride_t") and hasattr(vq_model, "down_t"):
        vq_stride = int(vq_model.stride_t ** vq_model.down_t)
    else:
        vq_stride = 1

    first_channel_num = None
    accumulator = None

    processed_batches = 0
    global_sequence_index = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if isinstance(batch, (tuple, list)):
            batch_x = batch[0]
            label = batch[1]
            padding_mask = batch[2] if len(batch) >= 3 else None
        elif isinstance(batch, dict):
            batch_x = batch.get("x", batch.get("features", batch.get("data")))
            label = batch.get("label", batch.get("labels", batch.get("y")))
            padding_mask = batch.get("padding_mask", None)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        if batch_x is None or label is None:
            raise RuntimeError(
                "batch_x or label is None. "
                "Expected dataloader output like (batch_x, label, padding_mask)."
            )

        batch_x = batch_x.float().to(device)
        label = label.long().view(-1).to(device)

        B, L_orig, C = batch_x.shape

        if first_channel_num is None:
            first_channel_num = C

            if channel_names is None:
                channel_names = [f"channel_{i}" for i in range(C)]

            if len(channel_names) != C:
                raise ValueError(
                    f"len(channel_names)={len(channel_names)} but input channel C={C}"
                )

            accumulator = PrimitiveAccumulator(
                num_codes=num_codes,
                num_channels=C,
                keep_examples_per_code=keep_examples_per_code,
            )

        x_pad, L_before_pad = pad_to_multiple(batch_x, multiple=vq_stride, pad_value=0.0)
        L_pad = x_pad.shape[1]

        valid_time_mask = pad_mask_to_len(
            padding_mask,
            B=B,
            L_orig=L_orig,
            L_pad=L_pad,
            device=device,
        )

        ids = vq_model.get_token_ids(x_pad).long().to(device)  # [B, P]

        if ids.dim() != 2:
            raise RuntimeError(f"vq_model.get_token_ids should return [B, P], got {tuple(ids.shape)}")

        B2, P = ids.shape

        if B2 != B:
            raise RuntimeError(f"Token batch mismatch: ids B={B2}, x B={B}")

        patch_len = int(math.ceil(L_pad / P))

        processed_batches += 1
        accumulator.total_sequences += B

        for b in range(B):
            y = int(label[b].item())

            if label_names is not None and 0 <= y < len(label_names):
                label_name = str(label_names[y])
            else:
                label_name = f"class_{y}"

            valid_ids_in_seq = []

            for p in range(P):
                st = p * patch_len
                ed = min((p + 1) * patch_len, L_pad)

                if st >= ed:
                    continue

                token_valid_ratio = valid_time_mask[b, st:ed].float().mean().item()

                if token_valid_ratio < float(min_valid_ratio_per_token):
                    valid_ids_in_seq.append(None)
                    continue

                cid = int(ids[b, p].item())

                if cid < 0 or cid >= num_codes:
                    valid_ids_in_seq.append(None)
                    continue

                valid_ids_in_seq.append(cid)

            for p in range(P):
                cid = valid_ids_in_seq[p]

                if cid is None:
                    continue

                st = p * patch_len
                ed = min((p + 1) * patch_len, L_pad)

                # Use only valid original positions inside this token.
                seg = x_pad[b, st:ed, :]
                seg_mask = valid_time_mask[b, st:ed]

                if seg_mask.sum() <= 0:
                    continue

                seg = seg[seg_mask]

                if seg.numel() == 0:
                    continue

                seg_stats = compute_segment_stats(seg)

                prev_id = None
                next_id = None

                for pp in range(p - 1, -1, -1):
                    if valid_ids_in_seq[pp] is not None:
                        prev_id = valid_ids_in_seq[pp]
                        break

                for pp in range(p + 1, P):
                    if valid_ids_in_seq[pp] is not None:
                        next_id = valid_ids_in_seq[pp]
                        break

                accumulator.add_occurrence(
                    cid=cid,
                    label_id=y,
                    label_name=label_name,
                    token_pos=p,
                    sequence_index=global_sequence_index + b,
                    segment_stats=seg_stats,
                    prev_id=prev_id,
                    next_id=next_id,
                )

        global_sequence_index += B

    if accumulator is None:
        raise RuntimeError("[PrimitiveProfile] No batches were processed.")

    if processed_batches == 0:
        raise RuntimeError("[PrimitiveProfile] processed_batches == 0.")

    if accumulator.total_tokens == 0:
        raise RuntimeError(
            "[PrimitiveProfile] total_tokens == 0. "
            "Check padding_mask, get_token_ids(), and min_valid_ratio_per_token."
        )

    used_codes = int((accumulator.code_count > 0).sum().item())

    print(
        f"[PrimitiveProfile] processed_batches={processed_batches}, "
        f"total_sequences={accumulator.total_sequences}, "
        f"total_tokens={accumulator.total_tokens}, "
        f"used_codes={used_codes}/{num_codes}"
    )

    # ------------------------------------------------------------
    # Build preliminary aggregate stats
    # ------------------------------------------------------------
    scalar_keys = [
        "energy",
        "rms",
        "mean_abs",
        "std",
        "peak_abs",
        "temporal_variation",
        "delta_energy",
        "jerk_abs",
        "periodicity",
        "spectral_entropy",
        "dominant_frequency_index",
        "zero_crossing_rate",
        "dominant_channel_energy_ratio",
    ]

    aggregate = {}

    for cid in range(num_codes):
        cnt = int(accumulator.code_count[cid].item())

        stats = {}

        for key in scalar_keys:
            mean, std = accumulator.scalar_mean_std(cid, key)

            if mean is not None:
                stats[f"{key}_mean"] = mean
                stats[f"{key}_std"] = std

        if cnt > 0 and accumulator.channel_ratio_count[cid].item() > 0:
            ch_mean = (
                accumulator.channel_ratio_sum[cid]
                / max(int(accumulator.channel_ratio_count[cid].item()), 1)
            )
            ch_mean_list = [safe_float(x) for x in ch_mean.tolist()]
            stats["channel_energy_ratio_mean"] = ch_mean_list

        if cnt > 0 and len(accumulator.dominant_channel_count[cid]) > 0:
            dom_ch, dom_ch_count = accumulator.dominant_channel_count[cid].most_common(1)[0]
            dom_ch = int(dom_ch)
            stats["dominant_channel"] = dom_ch
            stats["dominant_channel_name"] = str(channel_names[dom_ch])
            stats["dominant_channel_ratio"] = safe_float(dom_ch_count / cnt)
        else:
            stats["dominant_channel"] = None
            stats["dominant_channel_name"] = None
            stats["dominant_channel_ratio"] = 0.0

        label_distribution = build_distribution(
            accumulator.code_label_count[cid],
            total=cnt,
            label_names=label_names,
            topk=None,
        )

        if len(label_distribution) > 0:
            stats["label_purity"] = safe_float(label_distribution[0]["ratio"])
            stats["top_label_id"] = int(label_distribution[0]["label_id"])
            stats["top_label_name"] = str(label_distribution[0]["label_name"])
        else:
            stats["label_purity"] = 0.0
            stats["top_label_id"] = None
            stats["top_label_name"] = None

        stats["support_level"] = support_level(cnt)

        aggregate[cid] = {
            "count": cnt,
            "stats": stats,
            "label_distribution": label_distribution,
        }

    # ------------------------------------------------------------
    # Quantile thresholds across used codes
    # ------------------------------------------------------------
    def collect_mean(key):
        vals = []
        for cid in range(num_codes):
            if aggregate[cid]["count"] > 0:
                v = aggregate[cid]["stats"].get(f"{key}_mean", None)
                if v is not None:
                    vals.append(safe_float(v))
        return vals

    quantiles = {}

    for key in ["energy", "temporal_variation", "periodicity", "spectral_entropy"]:
        vals = collect_mean(key)
        quantiles[key] = (
            percentile(vals, 0.33, default=0.0),
            percentile(vals, 0.66, default=1.0),
        )

    # ------------------------------------------------------------
    # Final JSON
    # ------------------------------------------------------------
    profile: Dict[str, Any] = {}

    if include_meta:
        profile["_meta"] = {
            "type": "strong_primitive_profile",
            "num_codes": int(num_codes),
            "num_channels": int(first_channel_num),
            "channel_names": channel_names,
            "label_names": label_names,
            "vq_stride": int(vq_stride),
            "processed_batches": int(processed_batches),
            "total_sequences": int(accumulator.total_sequences),
            "total_tokens": int(accumulator.total_tokens),
            "used_codes": int(used_codes),
            "used_code_ratio": safe_float(used_codes / max(num_codes, 1)),
            "min_valid_ratio_per_token": float(min_valid_ratio_per_token),
            "quantile_thresholds": {
                k: [safe_float(v[0]), safe_float(v[1])] for k, v in quantiles.items()
            },
            "field_explanation": {
                "count": "Number of token occurrences assigned to this primitive.",
                "frequency": "count / total valid primitive tokens.",
                "label_distribution": "Activity-label distribution of samples where this primitive appears.",
                "label_purity": "Top-label ratio. Higher means more class-discriminative.",
                "energy_mean": "Average squared amplitude of the raw sensor segment.",
                "temporal_variation_mean": "Average first-order temporal change magnitude.",
                "periodicity_mean": "Autocorrelation-based local periodicity score.",
                "spectral_entropy_mean": "Normalized spectral entropy; lower means concentrated frequency.",
                "dominant_channel_name": "Sensor channel with largest energy contribution.",
                "case_study": "Natural-language summary for qualitative analysis.",
            },
        }

    for cid in range(num_codes):
        cnt = aggregate[cid]["count"]
        stats = aggregate[cid]["stats"]
        label_distribution = aggregate[cid]["label_distribution"]

        frequency = safe_float(cnt / max(accumulator.total_tokens, 1))

        prev_distribution = build_transition_distribution(
            accumulator.transition_prev_count[cid],
            total=cnt,
            topk=5,
        )

        next_distribution = build_transition_distribution(
            accumulator.transition_next_count[cid],
            total=cnt,
            topk=5,
        )

        description = make_description(
            count=cnt,
            stats=stats,
            label_distribution=label_distribution,
            quantiles=quantiles,
        )

        case_study = make_case_study_summary(
            primitive_id=cid,
            count=cnt,
            stats=stats,
            label_distribution=label_distribution,
        )

        semantic_score = compute_motion_semantic_score(
            count=cnt,
            total_tokens=accumulator.total_tokens,
            stats=stats,
        )

        profile[str(cid)] = {
            "id": int(cid),
            "count": int(cnt),
            "frequency": frequency,
            "semantic_score": semantic_score,
            "label_distribution": label_distribution,
            "transition_distribution": {
                "previous": prev_distribution,
                "next": next_distribution,
            },
            "stats": stats,
            "description": description,
            "case_study": case_study,
            "examples": accumulator.examples[cid],
        }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print(f"[PrimitiveProfile] Saved strong primitive profile to: {save_path}")

    return profile


def compute_motion_semantic_score(
    count: int,
    total_tokens: int,
    stats: Dict[str, Any],
):
    """
    A heuristic score for sorting/displaying primitives in analysis.

    High score means:
        - enough support
        - relatively pure activity association
        - clear dominant channel
        - non-trivial motion structure
    """
    if count <= 0:
        return 0.0

    support = min(math.log(count + 1.0) / math.log(max(total_tokens, 2.0)), 1.0)
    purity = safe_float(stats.get("label_purity", 0.0))
    channel_dom = safe_float(stats.get("dominant_channel_ratio", 0.0))
    energy = safe_float(stats.get("energy_mean", 0.0))
    periodicity = safe_float(stats.get("periodicity_mean", 0.0))
    temporal = safe_float(stats.get("temporal_variation_mean", 0.0))

    motion_strength = math.tanh(energy + temporal)
    structure = 0.5 * periodicity + 0.5 * channel_dom

    score = (
        0.25 * support
        + 0.35 * purity
        + 0.20 * structure
        + 0.20 * motion_strength
    )

    return safe_float(max(0.0, min(score, 1.0)))