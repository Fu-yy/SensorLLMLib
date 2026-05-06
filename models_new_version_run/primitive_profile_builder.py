import os
import json
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np


def _to_tensor(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    return torch.as_tensor(x, device=device)


def _unpack_batch(batch):
    """
    Compatible with common dataloader formats.

    Supported:
        (x, y)
        (x, y, ...)
        {"x": ..., "label": ...}
        {"features": ..., "labels": ...}
    """
    if isinstance(batch, dict):
        x = (
            batch.get("x", None)
            or batch.get("features", None)
            or batch.get("inputs", None)
            or batch.get("data", None)
        )
        y = (
            batch.get("y", None)
            or batch.get("label", None)
            or batch.get("labels", None)
            or batch.get("target", None)
        )
        padding_mask = batch.get("padding_mask", None)
        return x, y, padding_mask

    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Batch tuple/list should contain at least (x, y).")
        x = batch[0]
        y = batch[1]
        padding_mask = batch[2] if len(batch) >= 3 else None
        return x, y, padding_mask

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _segment_stats(seg: torch.Tensor) -> Dict[str, float]:
    """
    seg: [T, C]
    """
    seg = seg.float()

    abs_mean = seg.abs().mean().item()
    energy = (seg ** 2).mean().item()

    if seg.shape[0] > 1:
        delta = (seg[1:] - seg[:-1]).abs().mean().item()
    else:
        delta = 0.0

    channel_energy = (seg ** 2).mean(dim=0)
    dominant_channel = int(channel_energy.argmax().item())

    # Simple periodicity proxy: normalized lag-1 autocorrelation on magnitude.
    mag = torch.norm(seg, dim=-1)
    if mag.numel() > 2 and mag.std().item() > 1e-6:
        a = mag[:-1] - mag[:-1].mean()
        b = mag[1:] - mag[1:].mean()
        periodicity = (a * b).mean() / (a.std() * b.std() + 1e-6)
        periodicity = float(periodicity.clamp(-1, 1).item())
    else:
        periodicity = 0.0

    return {
        "abs_mean": float(abs_mean),
        "energy": float(energy),
        "delta": float(delta),
        "dominant_channel": int(dominant_channel),
        "periodicity": float(periodicity),
    }


def _describe_primitive(
    code_id: int,
    count: int,
    label_counter: Counter,
    label_names: Optional[List[str]],
    stats_mean: Dict[str, float],
) -> str:
    abs_mean = stats_mean.get("abs_mean", 0.0)
    energy = stats_mean.get("energy", 0.0)
    delta = stats_mean.get("delta", 0.0)
    periodicity = stats_mean.get("periodicity", 0.0)
    dominant_channel = int(stats_mean.get("dominant_channel", 0))

    if energy < 0.05:
        intensity = "very low-intensity or near-static"
    elif energy < 0.20:
        intensity = "low-intensity"
    elif energy < 0.60:
        intensity = "moderate-intensity"
    else:
        intensity = "high-intensity"

    if delta < 0.05:
        transition = "stable"
    elif delta < 0.20:
        transition = "smoothly varying"
    else:
        transition = "rapidly changing"

    if periodicity > 0.45:
        temporal = "periodic"
    elif periodicity > 0.15:
        temporal = "weakly periodic"
    else:
        temporal = "non-periodic"

    top_label_text = ""
    if len(label_counter) > 0:
        total = sum(label_counter.values())
        top_label, top_cnt = label_counter.most_common(1)[0]
        ratio = top_cnt / max(total, 1)

        if label_names is not None and 0 <= int(top_label) < len(label_names):
            label_str = label_names[int(top_label)]
        else:
            label_str = f"class {int(top_label)}"

        top_label_text = f", frequently associated with {label_str} ({ratio:.2f})"

    desc = (
        f"a {intensity}, {transition}, {temporal} motion primitive "
        f"with dominant response on channel {dominant_channel}"
        f"{top_label_text}"
    )

    return desc


@torch.no_grad()
def build_primitive_profile(
    vq_model,
    dataloader,
    save_path,
    device,
    num_codes=512,
    label_names=None,
    max_batches=None,
):
    import os
    import json
    import torch
    import numpy as np
    from collections import defaultdict, Counter

    if max_batches is not None:
        max_batches = int(max_batches)
        if max_batches <= 0:
            max_batches = None

    vq_model = vq_model.to(device)
    vq_model.eval()

    code_count = torch.zeros(num_codes, dtype=torch.long)
    code_label_count = [Counter() for _ in range(num_codes)]

    total_tokens = 0
    total_samples = 0
    processed_batches = 0

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

        batch_x = batch_x.float().to(device)
        label = label.long().view(-1).to(device)

        if padding_mask is not None:
            padding_mask = padding_mask.to(device)
            if padding_mask.dtype != torch.bool:
                padding_mask = padding_mask > 0

        # 优先使用 get_token_ids_with_mask
        if hasattr(vq_model, "get_token_ids_with_mask"):
            ids, valid_mask = vq_model.get_token_ids_with_mask(
                features=batch_x,
                padding_mask=padding_mask,
            )
        else:
            ids = vq_model.get_token_ids(batch_x)
            valid_mask = torch.ones_like(ids, dtype=torch.bool)

        ids = ids.long().detach().cpu()
        valid_mask = valid_mask.bool().detach().cpu()
        label_cpu = label.detach().cpu()

        B, P = ids.shape

        processed_batches += 1
        total_samples += B

        for b in range(B):
            y = int(label_cpu[b].item())

            valid_ids = ids[b][valid_mask[b]]

            if valid_ids.numel() == 0:
                continue

            for cid in valid_ids.tolist():
                cid = int(cid)
                if 0 <= cid < num_codes:
                    code_count[cid] += 1
                    code_label_count[cid][y] += 1
                    total_tokens += 1

    print(
        f"[PrimitiveProfile] processed_batches={processed_batches}, "
        f"total_samples={total_samples}, total_tokens={total_tokens}, "
        f"used_codes={(code_count > 0).sum().item()}/{num_codes}"
    )

    if processed_batches == 0:
        raise RuntimeError(
            "[PrimitiveProfile] processed_batches == 0. "
            "Check primitive_profile_max_batches. If it is -1, convert it to None."
        )

    if total_tokens == 0:
        raise RuntimeError(
            "[PrimitiveProfile] total_tokens == 0. "
            "No valid VQ tokens were collected. Check padding_mask convention and get_token_ids_with_mask()."
        )

    profile = {}

    for cid in range(num_codes):
        cnt = int(code_count[cid].item())
        freq = float(cnt / max(total_tokens, 1))

        label_dist = []
        if cnt > 0:
            for lid, c in code_label_count[cid].most_common():
                name = str(label_names[lid]) if label_names is not None and 0 <= lid < len(label_names) else f"class {lid}"
                label_dist.append({
                    "label_id": int(lid),
                    "label_name": name,
                    "count": int(c),
                    "ratio": float(c / cnt),
                })

        if cnt == 0:
            desc = "an unused motion primitive in the current training split"
        else:
            if len(label_dist) > 0:
                top = label_dist[0]
                desc = (
                    f"a motion primitive frequently associated with "
                    f"{top['label_name']} ({top['ratio']:.2f})"
                )
            else:
                desc = "a motion primitive observed in the current training split"

        profile[str(cid)] = {
            "id": int(cid),
            "count": cnt,
            "frequency": freq,
            "label_distribution": label_dist,
            "stats": {},
            "description": desc,
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print(f"[PrimitiveProfile] Saved primitive profile to: {save_path}")

    print("padding_mask shape:", padding_mask.shape)
    print("padding_mask true ratio:", padding_mask.float().mean().item())
    print("ids shape:", ids.shape)
    print("valid_mask true ratio:", valid_mask.float().mean().item())
    print("first ids:", ids[0][:20])
    print("first valid:", valid_mask[0][:20])

    return profile