import os
import re
import json
from collections import Counter
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml
except Exception:
    yaml = None

from transformers import AutoTokenizer, AutoModelForCausalLM
from models_new_version_run.VQ_VAE import IMU_VQ_Model


# ============================================================
# Basic helpers
# ============================================================
def infer_label_from_text(text: str, label_names: List[str]):
    if text is None:
        return None

    t = str(text).lower()

    # 长标签优先，避免 Walking 抢 Walking upstairs / Walking downstairs
    items = sorted(
        list(enumerate(label_names)),
        key=lambda x: len(x[1]),
        reverse=True,
    )

    for idx, name in items:
        if str(name).lower() in t:
            return int(idx)

    return None
def pad_to_multiple(x: torch.Tensor, multiple: int, pad_value: float = 0.0):
    if x.dim() != 3:
        raise ValueError(f"x should be [B, L, C], got {tuple(x.shape)}")

    B, L, C = x.shape
    L_pad = ((L + multiple - 1) // multiple) * multiple

    if L_pad == L:
        return x, L

    pad_len = L_pad - L
    pad = x.new_full((B, pad_len, C), pad_value)
    return torch.cat([x, pad], dim=1), L


def get_label_names_from_cfg(ds_cfg: Dict[str, Any], num_class: int):
    if isinstance(ds_cfg, dict):
        if "label_names" in ds_cfg and ds_cfg["label_names"] is not None:
            names = [str(x) for x in ds_cfg["label_names"]]
            if len(names) != int(num_class):
                raise ValueError(
                    f"len(label_names)={len(names)} != num_class={num_class}"
                )
            return names

        if "id2label" in ds_cfg and ds_cfg["id2label"] is not None:
            raw = ds_cfg["id2label"]
            id2label = {int(k): str(v) for k, v in raw.items()}

            missing = [i for i in range(int(num_class)) if i not in id2label]
            if len(missing) > 0:
                raise ValueError(
                    f"id2label missing ids: {missing}. "
                    f"Available keys: {sorted(id2label.keys())}"
                )

            return [id2label[i] for i in range(int(num_class))]

    return [f"class_{i}" for i in range(int(num_class))]


def safe_json_load(path: str):
    if path is None:
        raise ValueError("profile_path is None.")

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compress_repeated_ids(seq: List[int]):
    if len(seq) == 0:
        return []

    out = []
    cur = seq[0]
    cnt = 1

    for x in seq[1:]:
        if x == cur:
            cnt += 1
        else:
            out.append((cur, cnt))
            cur = x
            cnt = 1

    out.append((cur, cnt))
    return out


def format_compressed_sequence(seq: List[int]):
    compressed = compress_repeated_ids(seq)
    parts = []

    for pid, cnt in compressed:
        if cnt > 1:
            parts.append(f"Primitive {pid} repeated {cnt} times")
        else:
            parts.append(f"Primitive {pid}")

    return " -> ".join(parts)


def extract_json_from_response(text: str):
    """
    Robustly parse JSON from LLM response.

    Supports:
        1. pure JSON
        2. ```json fenced JSON
        3. extra text before/after JSON
    """
    text = str(text).strip()

    if len(text) == 0:
        return None

    # 1. Direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2. Markdown fenced block
    if "```" in text:
        parts = text.replace("```json", "```").split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                try:
                    return json.loads(part)
                except Exception:
                    pass

    # 3. First { to last }
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 4. Conservative object candidates
    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.S)
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue

    return None


# ============================================================
# Prompt-LLM HAR Model
# ============================================================

class AlignmentModel(nn.Module):
    """
    Prompt-based LLM HAR model.

    This model is inference-only:
        IMU -> VQ primitive ids -> primitive-profile prompt -> LLM JSON output -> label logits

    Compatible with Exp_Alignment_Classification:
        logits, probs = model.classify(batch_x, padding_mask)

    Important:
        This model does not support gradient-based training.
        Use it for prompt evaluation / case study / qualitative analysis.
    """

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device = args.device

        # This model is checkpoint-free / inference-only.
        self.is_prompt_llm_model = True
        self.requires_checkpoint = False
        self.inference_only = True

        self.dataset_key = str(
            getattr(args, "dataset_key", getattr(args, "data", "mhealth"))
        ).lower()

        # ============================================================
        # Dataset config
        # ============================================================
        if hasattr(args, "ds_cfg") and isinstance(args.ds_cfg, dict):
            self.ds_cfg = args.ds_cfg
        else:
            self.ds_cfg = self._load_ds_cfg_from_yaml(args)

        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 15)))
        self.num_class = int(self.ds_cfg.get("num_labels", getattr(args, "num_class", 12)))
        self.seq_len_orig = int(getattr(args, "seq_len", 200))

        self.label_names = get_label_names_from_cfg(
            self.ds_cfg,
            num_class=self.num_class,
        )

        print(f"[PrimitivePromptLLM] dataset_key={self.dataset_key}")
        print(f"[PrimitivePromptLLM] channel_num={self.C}, num_class={self.num_class}")
        print(f"[PrimitivePromptLLM] label_names={self.label_names}")

        # ============================================================
        # Load frozen VQ-VAE
        # ============================================================
        self.vq_net = IMU_VQ_Model(args)

        self.qua_path = self._resolve_vqvae_ckpt_dir(args)
        vq_ckpt_path = os.path.join(self.qua_path, "best_wrapper.pth")

        if not os.path.exists(vq_ckpt_path):
            raise FileNotFoundError(f"VQ-VAE checkpoint not found: {vq_ckpt_path}")

        print(f"[PrimitivePromptLLM] Loading VQ-VAE from: {vq_ckpt_path}")

        vq_state = torch.load(vq_ckpt_path, map_location="cpu")

        if isinstance(vq_state, dict) and "state_dict" in vq_state:
            vq_state = vq_state["state_dict"]
        elif isinstance(vq_state, dict) and "model" in vq_state:
            vq_state = vq_state["model"]
        elif isinstance(vq_state, dict) and "model_state_dict" in vq_state:
            vq_state = vq_state["model_state_dict"]

        missing, unexpected = self.vq_net.load_state_dict(vq_state, strict=False)
        print(f"[PrimitivePromptLLM] VQ missing={missing[:10]}")
        print(f"[PrimitivePromptLLM] VQ unexpected={unexpected[:10]}")

        self.vq_net.to(self.device)
        self.vq_net.eval()

        for p in self.vq_net.parameters():
            p.requires_grad = False

        self.vq_stride = int(self.vq_net.stride_t ** self.vq_net.down_t)

        self.seq_len_pad = (
            (self.seq_len_orig + self.vq_stride - 1) // self.vq_stride
        ) * self.vq_stride

        self.P = self.seq_len_pad // self.vq_stride

        print(
            f"[PrimitivePromptLLM] Orig={self.seq_len_orig}, "
            f"Stride={self.vq_stride}, Padded={self.seq_len_pad}, Patches={self.P}"
        )

        self.num_primitives = int(getattr(self.vq_net, "code_num", 512))
        self.mask_token_id = self.num_primitives

        # ============================================================
        # Profile
        # ============================================================
        profile_path = self._resolve_profile_path(args)

        print(f"[PrimitivePromptLLM] Using primitive profile: {profile_path}")

        self.profile_path = profile_path
        self.profile = safe_json_load(profile_path)

        # ============================================================
        # LLM
        # ============================================================
        llm_path = getattr(args, "llama_name", None)

        if llm_path is None:
            llm_path = getattr(args, "llm_path", None)

        if llm_path is None:
            raise ValueError("Please set args.llama_name or args.llm_path.")

        print(f"[PrimitivePromptLLM] Loading LLM from: {llm_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            use_fast=False,
            trust_remote_code=True,
        )

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device).eval()

        # Robust pad token handling.
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                self.llm.resize_token_embeddings(len(self.tokenizer))

        if hasattr(self.llm, "config"):
            self.llm.config.use_cache = True
            self.llm.config.pad_token_id = self.tokenizer.pad_token_id

        for p in self.llm.parameters():
            p.requires_grad = False

        # ============================================================
        # Prompt settings
        # ============================================================
        self.max_primitives_in_prompt = int(
            getattr(args, "prompt_max_primitives", 16)
        )

        self.max_new_tokens = int(
            getattr(args, "prompt_max_new_tokens", 128)
        )

        self.save_prompt = bool(
            int(getattr(args, "prompt_save_text", 0))
        )

        self.prompt_mode = str(
            getattr(args, "prompt_mode", "no_label")
        )

        self.default_confidence = str(
            getattr(args, "prompt_default_confidence", "medium")
        )

        self.prompt_max_length = int(
            getattr(args, "prompt_max_length", 4096)
        )

        self.llm_failures = 0
        self.last_prompt_info: List[Dict[str, Any]] = []

    # ============================================================
    # Config loading
    # ============================================================

    def _load_ds_cfg_from_yaml(self, args):
        ts_yaml = getattr(args, "ts_backbone_yaml", None)

        if ts_yaml is None:
            raise ValueError("args.ds_cfg is not provided and ts_backbone_yaml is None.")

        if yaml is None:
            raise ImportError("pyyaml is required to load ts_backbone_yaml.")

        if os.path.exists(ts_yaml):
            config_path = ts_yaml
        else:
            project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(project_path, "configs", ts_yaml)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"ts_backbone_yaml not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg_all = yaml.safe_load(f)

        if self.dataset_key not in cfg_all:
            raise KeyError(f"{self.dataset_key} not found in {config_path}")

        return cfg_all[self.dataset_key]

    def _resolve_vqvae_ckpt_dir(self, args):
        vqvae_key = getattr(args, "vqvae_path", None)

        if vqvae_key is not None:
            vqvae_key = str(vqvae_key)

            if isinstance(self.ds_cfg, dict) and vqvae_key in self.ds_cfg:
                return self.ds_cfg[vqvae_key]

            if os.path.exists(vqvae_key):
                return vqvae_key

        qua_path = getattr(args, "vqvae_ckpt_dir", None)

        if qua_path is not None and os.path.exists(str(qua_path)):
            return str(qua_path)

        raise ValueError(
            "VQ-VAE checkpoint directory is not provided. "
            "Set --vqvae_path as a key in ds_cfg, e.g. all_path, "
            "or set --vqvae_ckpt_dir."
        )

    def _resolve_profile_path(self, args):
        # 1. Explicit profile path.
        profile_path = getattr(args, "primitive_profile_path", None)

        if profile_path is not None and os.path.exists(str(profile_path)):
            return str(profile_path)

        # 2. Prefer profile generated by exp.
        no_label = getattr(args, "primitive_profile_no_label_path", None)

        if no_label is not None and os.path.exists(str(no_label)):
            return str(no_label)

        with_label = getattr(args, "primitive_profile_with_label_path", None)

        if with_label is not None and os.path.exists(str(with_label)):
            return str(with_label)

        # 3. Search under VQ ckpt directory.
        candidates = [
            os.path.join(self.qua_path, "primitive_profiles", "primitive_profile_no_label.json"),
            os.path.join(self.qua_path, "primitive_profiles", "primitive_profile_with_label.json"),
            os.path.join(self.qua_path, "primitive_profile_no_label.json"),
            os.path.join(self.qua_path, "primitive_profile_with_label.json"),
            os.path.join(self.qua_path, "primitive_profile_strong.json"),
            os.path.join(self.qua_path, "primitive_profile.json"),
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            "No primitive profile found. Tried:\n" + "\n".join(candidates)
        )

    # ============================================================
    # Primitive extraction
    # ============================================================

    @torch.no_grad()
    def _extract_primitive_sequences(self, x_imu, padding_mask=None):
        if not torch.is_tensor(x_imu):
            x_imu = torch.as_tensor(x_imu)

        x_imu = x_imu.to(self.device).float()

        B, L_orig, C = x_imu.shape

        if C != self.C:
            raise ValueError(f"Input channel C={C}, but dataset expects C={self.C}.")

        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device).bool()

        x_pad, _ = pad_to_multiple(x_imu, self.vq_stride, pad_value=0.0)

        if x_pad.shape[1] != self.seq_len_pad:
            if x_pad.shape[1] > self.seq_len_pad:
                x_pad = x_pad[:, :self.seq_len_pad, :]

                if padding_mask is not None:
                    padding_mask = padding_mask[:, :self.seq_len_pad]
            else:
                pad_len = self.seq_len_pad - x_pad.shape[1]
                pad = x_pad.new_zeros(B, pad_len, C)
                x_pad = torch.cat([x_pad, pad], dim=1)

                if padding_mask is not None:
                    pad_m = torch.zeros((B, pad_len), device=self.device, dtype=torch.bool)
                    padding_mask = torch.cat([padding_mask, pad_m], dim=1)

        gt_ids, valid_mask = self.vq_net.get_token_ids_with_mask(
            features=x_pad,
            padding_mask=padding_mask,
        )

        gt_ids = gt_ids.to(self.device).long()
        valid_mask = valid_mask.to(self.device).bool()

        if gt_ids.shape[1] != self.P:
            if gt_ids.shape[1] > self.P:
                gt_ids = gt_ids[:, :self.P]
                valid_mask = valid_mask[:, :self.P]
            else:
                raise ValueError(f"VQ tokens {gt_ids.shape[1]} < expected P={self.P}")

        seqs = []

        for b in range(B):
            seq = gt_ids[b][valid_mask[b]].detach().cpu().tolist()
            seqs.append([int(x) for x in seq])

        return seqs, gt_ids, valid_mask

    # ============================================================
    # Prompt building
    # ============================================================

    def _select_primitives_for_prompt(self, seq: List[int]):
        counter = Counter(seq)
        selected = []

        for pid, _ in counter.most_common(self.max_primitives_in_prompt):
            key = str(pid)
            if key in self.profile:
                selected.append(int(pid))

        return selected

    def _primitive_description_line(self, pid: int):
        item = self.profile.get(str(pid), None)

        if item is None:
            return f"Primitive {pid}: no reliable semantic profile is available."

        if isinstance(item, str):
            return f"Primitive {pid}: {item}"

        desc = str(item.get("description", f"motion primitive {pid}"))

        stats = item.get("stats", {})
        trans = item.get("transition_distribution", {})

        stat_parts = []

        if isinstance(stats, dict):
            if "energy_mean" in stats:
                stat_parts.append(f"energy={float(stats['energy_mean']):.4f}")
            if "temporal_variation_mean" in stats:
                stat_parts.append(f"temporal_variation={float(stats['temporal_variation_mean']):.4f}")
            if "periodicity_mean" in stats:
                stat_parts.append(f"periodicity={float(stats['periodicity_mean']):.4f}")
            if "spectral_entropy_mean" in stats:
                stat_parts.append(f"spectral_entropy={float(stats['spectral_entropy_mean']):.4f}")
            if "dominant_channel_name" in stats:
                stat_parts.append(f"dominant_channel={stats['dominant_channel_name']}")

        stat_text = ""
        if len(stat_parts) > 0:
            stat_text = " Key statistics: " + ", ".join(stat_parts) + "."

        trans_text = ""

        if isinstance(trans, dict):
            prevs = trans.get("previous", [])[:3]
            nexts = trans.get("next", [])[:3]

            prev_text = ", ".join([
                f"{x.get('code_id')}({float(x.get('ratio', 0.0)):.2f})"
                for x in prevs
            ])

            next_text = ", ".join([
                f"{x.get('code_id')}({float(x.get('ratio', 0.0)):.2f})"
                for x in nexts
            ])

            trans_parts = []

            if prev_text:
                trans_parts.append(f"common previous: {prev_text}")
            if next_text:
                trans_parts.append(f"common next: {next_text}")

            if len(trans_parts) > 0:
                trans_text = " Transition context: " + "; ".join(trans_parts) + "."

        return f"Primitive {pid}: {desc}.{stat_text}{trans_text}"

    def build_prompt(self, primitive_seq: List[int]):
        primitive_seq = [int(x) for x in primitive_seq]

        compressed_text = format_compressed_sequence(primitive_seq)

        selected_pids = self._select_primitives_for_prompt(primitive_seq)

        desc_lines = []

        for pid in selected_pids:
            desc_lines.append(self._primitive_description_line(pid))

        if len(desc_lines) == 0:
            desc_lines.append("No reliable primitive descriptions are available for this sample.")

        label_text = "\n".join([
            f"{i}: {name}" for i, name in enumerate(self.label_names)
        ])

        if self.prompt_mode == "with_label":
            note = (
                "Note: some primitive descriptions may include empirical activity associations. "
                "Use this setting only for qualitative explanation."
            )
        else:
            note = (
                "Important: use only motion properties, temporal order, repetition pattern, "
                "and transition context. Do not assume hidden label association."
            )

        prompt = f"""
You are an expert in wearable sensor-based human activity recognition.

A wearable IMU signal window has been converted into a sequence of discrete motion primitives.
Each primitive is a reusable local motion pattern learned by a VQ-VAE tokenizer.

Primitive sequence:
{primitive_seq}

Compressed primitive sequence:
{compressed_text}

Primitive descriptions:
{chr(10).join(desc_lines)}

Candidate activity labels:
{label_text}

{note}

Task:
You must choose exactly one activity label from the candidate list.

Output rules:
1. Output JSON only.
2. Do not repeat the prompt.
3. Do not continue the primitive sequence.
4. Do not output Markdown.
5. Do not output explanations outside JSON.
6. "predicted_label_id" must be an integer from 0 to {len(self.label_names) - 1}.
7. "predicted_label_name" must be exactly one of the candidate activity labels.

Now output only this JSON object:
{{
  "predicted_label_id": 0,
  "predicted_label_name": "{self.label_names[0]}",
  "confidence": "low",
  "reason": "brief reason based on the primitive sequence"
}}
""".strip()

        return prompt

    # ============================================================
    # LLM generation
    # ============================================================

    @torch.no_grad()
    def _format_generation_text(self, prompt: str):
        system_text = (
            "You are a careful expert in wearable sensor-based human activity recognition. "
            "You must output only one valid JSON object. "
            "Do not repeat the user prompt. "
            "Do not continue the prompt. "
            "Do not output explanations outside JSON."
        )

        has_valid_chat_template = (
                hasattr(self.tokenizer, "apply_chat_template")
                and getattr(self.tokenizer, "chat_template", None) is not None
                and str(getattr(self.tokenizer, "chat_template", "")).strip() != ""
        )

        if has_valid_chat_template:
            messages = [
                {
                    "role": "system",
                    "content": system_text,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            return text

        # Fallback only for non-chat models.
        # This is weaker and may cause base models to repeat the prompt.
        return (
            f"{system_text}\n\n"
            f"### User\n"
            f"{prompt}\n\n"
            f"### Assistant\n"
        )

    @torch.no_grad()
    def _generate_one(self, prompt: str):
        text = self._format_generation_text(prompt)

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.prompt_max_length,
        ).to(self.device)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        # Llama-3 eot token, if available
        eos_token_ids = []

        if self.tokenizer.eos_token_id is not None:
            eos_token_ids.append(self.tokenizer.eos_token_id)

        try:
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in eos_token_ids:
                eos_token_ids.append(eot_id)
        except Exception:
            pass

        if len(eos_token_ids) == 0:
            eos_token_ids = None
        elif len(eos_token_ids) == 1:
            eos_token_ids = eos_token_ids[0]

        try:
            generated = self.llm.generate(
                **encoded,
                max_new_tokens=int(self.max_new_tokens),
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_ids,
                repetition_penalty=1.05,
            )

            input_len = encoded["input_ids"].shape[1]

            response = self.tokenizer.decode(
                generated[0][input_len:],
                skip_special_tokens=True,
            ).strip()

            # ------------------------------------------------------------
            # Detect prompt-copy failure.
            # ------------------------------------------------------------
            bad_prefixes = [
                "You are an expert in wearable sensor-based human activity recognition",
                "A wearable IMU signal window has been converted",
                "Primitive sequence:",
                "Compressed primitive sequence:",
            ]

            if any(response.startswith(x) for x in bad_prefixes):
                return json.dumps(
                    {
                        "predicted_label_id": -1,
                        "predicted_label_name": "prompt_copy_failure",
                        "confidence": "low",
                        "reason": "LLM repeated the prompt instead of answering. This usually means chat_template is missing or the model is not instruction-following."
                    },
                    ensure_ascii=False,
                )

            return response

        except Exception as e:
            self.llm_failures += 1

            return json.dumps(
                {
                    "predicted_label_id": -1,
                    "predicted_label_name": "generation_failed",
                    "confidence": "low",
                    "reason": f"LLM generation failed: {type(e).__name__}: {str(e)}",
                },
                ensure_ascii=False,
            )

    def _parsed_to_logits(self, parsed: Optional[Dict[str, Any]]):
        logits = torch.zeros(self.num_class, device=self.device)

        pred_id = None
        confidence = self.default_confidence

        if parsed is not None:
            try:
                pred_id = int(parsed.get("predicted_label_id"))
                confidence = str(parsed.get("confidence", self.default_confidence)).lower()
            except Exception:
                pred_id = None

        if pred_id is None or pred_id < 0 or pred_id >= self.num_class:
            return logits, None, confidence

        if confidence == "high":
            score = 8.0
        elif confidence == "low":
            score = 3.0
        else:
            score = 5.0

        logits[pred_id] = score

        return logits, pred_id, confidence

    # ============================================================
    # Public classify interface
    # ============================================================

    @torch.no_grad()
    def classify(self, x_imu, padding_mask=None):
        """
        Compatible with Exp_Alignment_Classification._forward_classify().

        Returns:
            logits: [B, num_class]
            probs:  [B, num_class]
        """
        self.eval()

        seqs, gt_ids, valid_mask = self._extract_primitive_sequences(
            x_imu=x_imu,
            padding_mask=padding_mask,
        )

        logits_list = []
        prompt_info = []

        for b, seq in enumerate(seqs):
            prompt = self.build_prompt(seq)
            response = self._generate_one(prompt)
            parsed = extract_json_from_response(response)

            logits_b, pred_id, confidence = self._parsed_to_logits(parsed)

            # Fallback: if JSON parsing fails, try to infer label name from raw response.
            if pred_id is None:
                fallback_id = infer_label_from_text(response, self.label_names)

                if fallback_id is not None:
                    pred_id = int(fallback_id)
                    confidence = "low"

                    logits_b = torch.zeros(self.num_class, device=self.device)
                    logits_b[pred_id] = 3.0

                    parsed = {
                        "predicted_label_id": pred_id,
                        "predicted_label_name": self.label_names[pred_id],
                        "confidence": confidence,
                        "reason": "Recovered by label-name fallback parsing from raw LLM response."
                    }

            logits_list.append(logits_b)

            info = {
                "index_in_batch": int(b),
                "primitive_seq": seq,
                "prompt": prompt if self.save_prompt else None,
                "response": response,
                "parsed": parsed,
                "predicted_label_id": pred_id,
                "confidence": confidence,
            }

            prompt_info.append(info)

        logits = torch.stack(logits_list, dim=0)
        probs = F.softmax(logits, dim=-1)

        self.last_prompt_info = prompt_info

        return logits, probs

    def forward(self, x_imu, padding_mask=None, mode=None, labels=None, **kwargs):
        logits, probs = self.classify(x_imu, padding_mask=padding_mask)

        if mode in ["return_tuple", "with_probs"]:
            return logits, probs

        return logits

    # ============================================================
    # Dummy save/load for exp compatibility
    # ============================================================

    def save_wrapper(self, path):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        payload = {
            "model_type": "PrimitivePromptLLM",
            "profile_path": self.profile_path,
            "dataset_key": self.dataset_key,
            "label_names": self.label_names,
            "num_class": self.num_class,
        }

        torch.save(payload, path)

    def load_wrapper(self, path, map_location="cpu"):
        if os.path.exists(path):
            print(f"[PrimitivePromptLLM] load_wrapper ignored for prompt-only model: {path}")