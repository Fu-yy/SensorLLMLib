import os
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import yaml  # pyyaml
except Exception:
    yaml = None


class Model(nn.Module):
    """
    LLaMA wrapper (reconstruction pretrain style):
      1) Randomly mask sensor values in batch_x (continuous window).
      2) Build a prompt containing C placeholders: <TS_0> ... <TS_{C-1}>.
      3) Replace each placeholder token embedding with a channel-wise sensor embedding [B,C,H].
      4) Run LLaMA forward using inputs_embeds.
      5) Return:
         - out: LLM outputs
         - mask: [B,L,C] bool (True = masked)
         - x_masked: [B,L,C] masked input
         - meta: dict with useful debug info

    Input:
      batch_x: [B, L, C]
      padding_mask: [B, L] (optional; for your fixed windows it's all ones; we don't need it here)
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # -----------------------
        # 0) Optional YAML config (SensorLLM-style)
        # -----------------------
        # You can pass:
        #   --ts_backbone_yaml=.../ts_backbone.yaml
        #   --dataset_key=mhealth   (or use args.data)
        self.ds_cfg: Dict[str, Any] = {}
        self.dataset_key = str(getattr(args, "dataset_key", getattr(args, "data", "mhealth"))).lower()
        self.ts_backbone_yaml = getattr(args, "ts_backbone_yaml", None)

        if self.ts_backbone_yaml is not None:
            if yaml is None:
                raise ImportError("pyyaml not installed, but ts_backbone_yaml is set. Please `pip install pyyaml`.")
            self.ds_cfg = self._load_dataset_cfg(self.ts_backbone_yaml, self.dataset_key)

        # -----------------------
        # 1) Basic config
        # -----------------------
        # channel count: prefer YAML (B), fallback to args.enc_in
        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 15)))
        self.sample_rate = int(self.ds_cfg.get("sample_rate", 0))
        self.channel_names: List[str] = self.ds_cfg.get("channel_names", None)
        if self.channel_names is None:
            self.channel_names = [f"ch{i}" for i in range(self.C)]
        else:
            if len(self.channel_names) != self.C:
                raise ValueError(f"channel_names length {len(self.channel_names)} != channel_num {self.C}")

        self.mask_rate = float(getattr(args, "mask_rate", 0.3))
        if not (0.0 <= self.mask_rate <= 1.0):
            raise ValueError(f"mask_rate must be in [0,1], got {self.mask_rate}")

        # prompt + tokenization
        self.max_prompt_len = int(getattr(args, "max_prompt_len", 512))
        if self.max_prompt_len < 32:
            raise ValueError(f"max_prompt_len too small: {self.max_prompt_len}")

        # LLaMA path/name
        self.llama_name = getattr(args, "llama_name", r"D:\fuy\MyCode\SensorLLM\Llama-3.2-1B")

        # whether to return hidden states
        self.output_hidden_states = bool(getattr(args, "output_hidden_states", True))

        # -----------------------
        # 2) Tokenizer + LLM
        # -----------------------
        self.tokenizer = AutoTokenizer.from_pretrained(self.llama_name, use_fast=False)
        # LLaMA often has no pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llama_name,
            torch_dtype=getattr(torch, getattr(args, "llm_dtype", "float16"), torch.float16),
        )
        # Ensure config matches
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id

        # -----------------------
        # 3) Add special tokens + resize
        # -----------------------
        # NOTE: This IS required because we use new tokens like <TS_0>.
        self.ts_tokens = [f"<TS_{i}>" for i in range(self.C)]
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": self.ts_tokens})
        if added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))

        # Cache ids for fast matching during injection
        self.ts_token_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in self.ts_tokens]
        if any(tid is None or tid < 0 for tid in self.ts_token_ids):
            raise RuntimeError("Failed to convert some <TS_i> tokens to ids. Check tokenizer special tokens.")

        # -----------------------
        # 4) Sensor encoder -> channel-wise embedding [B,C,H]
        # -----------------------
        llama_hidden = int(self.llm.config.hidden_size)

        # A simple baseline encoder:
        #   mean pool over time per channel -> [B,C]
        #   then project scalar -> H
        self.sensor_proj = nn.Linear(1, llama_hidden)
        self.channel_id = nn.Embedding(self.C, llama_hidden)

        # -----------------------
        # 5) Prompt text cache (template is constant for this model instance)
        # -----------------------
        self._prompt_text_cache: Optional[str] = None

        # -----------------------
        # 6) Mask description settings (optional, keep prompt stable)
        # -----------------------
        # If True, we add a SHORT masked summary string to prompt, but keep it bounded.
        self.include_mask_summary = bool(getattr(args, "include_mask_summary", True))
        self.max_mask_spans_in_prompt = int(getattr(args, "max_mask_spans_in_prompt", 8))  # keep small

    # ============================================================
    # YAML loader
    # ============================================================
    @staticmethod
    def _load_dataset_cfg(yaml_path: str, dataset_key: str) -> Dict[str, Any]:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"ts_backbone_yaml not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg_all = yaml.safe_load(f)
        if dataset_key not in cfg_all:
            raise KeyError(f"dataset '{dataset_key}' not found in {yaml_path}")
        return cfg_all[dataset_key]

    # ============================================================
    # 1) Masking
    # ============================================================
    def _random_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B,L,C]
        Returns:
          x_masked: [B,L,C] (masked values set to 0)
          mask: [B,L,C] bool (True where masked)
        """
        if x.dim() != 3:
            raise ValueError(f"batch_x must be 3D [B,L,C], got {tuple(x.shape)}")
        B, L, C = x.shape
        if C != self.C:
            raise ValueError(f"Input C={C} != model C={self.C}. Check enc_in / dataset config.")
        device = x.device

        # independent element-wise mask (you can later change to block mask)
        mask = torch.rand(B, L, C, device=device) < self.mask_rate
        x_masked = x.clone()
        x_masked = x_masked.masked_fill(mask, 0.0)
        return x_masked, mask

    # ============================================================
    # 2) Sensor -> [B,C,H]
    # ============================================================
    def _encode_sensor(self, x_masked: torch.Tensor) -> torch.Tensor:
        """
        x_masked: [B,L,C]
        output: sensor_embeds [B,C,H]
        """
        B, L, C = x_masked.shape

        # mean over time per channel: [B,C]
        pooled = x_masked.mean(dim=1)  # [B,C]
        pooled = pooled.unsqueeze(-1)  # [B,C,1]
        sensor_embeds = self.sensor_proj(pooled)  # [B,C,H]

        # add channel identity embedding
        ch = torch.arange(C, device=x_masked.device).unsqueeze(0).expand(B, C)  # [B,C]
        sensor_embeds = sensor_embeds + self.channel_id(ch)  # [B,C,H]
        return sensor_embeds

    # ============================================================
    # 3) Prompt building (B: semantic channel names + optional mask summary)
    # ============================================================
    def _build_prompt_text(self, mask_summary: Optional[str] = None) -> str:
        """
        Build ONE prompt string. We keep it stable/short.
        """
        lines = []
        lines.append("You are a sensor time-series reconstruction assistant.")
        lines.append(f"Dataset: {self.dataset_key}.")
        if self.sample_rate > 0:
            lines.append(f"Sampling rate: {self.sample_rate} Hz.")
        lines.append(f"The input contains {self.C} sensor channels. Each channel is represented by a placeholder token.")
        lines.append("Task: reconstruct the missing values for the masked time series.")
        if mask_summary:
            lines.append("")
            lines.append(f"Masked summary: {mask_summary}")
        lines.append("")
        lines.append("Channels:")
        for i, name in enumerate(self.channel_names):
            lines.append(f"{name}: <TS_{i}>")
        lines.append("")
        lines.append("Output: provide the reconstructed values for the masked positions.")
        return "\n".join(lines)

    def _mask_to_summary(self, mask: torch.Tensor) -> str:
        """
        Convert mask [B,L,C] into a short summary string for PROMPT.
        Must be bounded length (do NOT dump full indices).
        Strategy:
          - For each sample, summarize a few (channel, time-span) masked runs.
        """
        # We'll only use sample 0's summary to keep prompt identical across batch? (Better: per-sample prompt)
        # Here we do per-sample prompts to be faithful, but keep it bounded anyway.
        # This function returns summary for ONE sample mask: [L,C] bool.
        if mask.dim() != 2:
            raise ValueError(f"mask_to_summary expects [L,C], got {tuple(mask.shape)}")
        L, C = mask.shape
        spans = []
        # collect first few spans across channels
        for c in range(C):
            m = mask[:, c]  # [L]
            # find runs of True
            in_run = False
            s = 0
            for t in range(L):
                if m[t].item() and not in_run:
                    in_run = True
                    s = t
                if in_run and (not m[t].item() or t == L - 1):
                    e = t if m[t].item() else (t - 1)
                    spans.append((c, s, e))
                    in_run = False
                    if len(spans) >= self.max_mask_spans_in_prompt:
                        break
            if len(spans) >= self.max_mask_spans_in_prompt:
                break

        if not spans:
            return "none"

        # format: chName[t0-t1], ...
        parts = []
        for (c, s, e) in spans:
            cname = self.channel_names[c] if c < len(self.channel_names) else f"ch{c}"
            if s == e:
                parts.append(f"{cname}[{s}]")
            else:
                parts.append(f"{cname}[{s}-{e}]")
        return ", ".join(parts)

    @torch.no_grad()
    def _tokenize_prompts(self, prompts: List[str], device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        prompts: list length B
        return:
          input_ids: [B,T]
          attn_mask: [B,T]
        """
        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_len,
            return_tensors="pt",
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    # ============================================================
    # 4) Inject sensor embeddings into placeholder positions
    # ============================================================
    def _inject_sensor_embeds(self, input_ids: torch.Tensor, sensor_embeds: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B,T]
        sensor_embeds: [B,C,H]
        return inputs_embeds: [B,T,H]
        """
        B, T = input_ids.shape
        B2, C, H = sensor_embeds.shape
        if B2 != B:
            raise ValueError(f"sensor_embeds batch {B2} != input_ids batch {B}")
        if C != self.C:
            raise ValueError(f"sensor_embeds C={C} != model C={self.C}")

        embed_layer = self.llm.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)  # [B,T,H]

        # For each channel placeholder, replace exactly one position per sample.
        # If prompt is accidentally truncated so that <TS_i> disappears, we skip safely but warn via meta.
        for i, ts_id in enumerate(self.ts_token_ids):
            pos = (input_ids == ts_id)  # [B,T]
            if pos.any():
                # We expect exactly one per sample. Enforce: if not, still handle.
                # Gather indices where pos==True and assign by sample.
                # We'll do per-sample assignment robustly.
                for b in range(B):
                    idxs = torch.nonzero(pos[b], as_tuple=False).flatten()
                    if idxs.numel() == 0:
                        continue
                    # if multiple occurrences, replace all with same channel embedding
                    inputs_embeds[b, idxs, :] = sensor_embeds[b, i, :].unsqueeze(0).expand(idxs.numel(), H)

        return inputs_embeds

    # ============================================================
    # Forward
    # ============================================================
    def forward(self, batch_x, padding_mask=None, *args, **kwargs):
        """
        batch_x: [B,L,C]
        padding_mask: [B,L] (unused here; prompt attention uses LLM tokenizer attention_mask)
        returns:
          out, mask, x_masked, meta
        """
        if not torch.is_tensor(batch_x):
            batch_x = torch.as_tensor(batch_x)

        B, L, C = batch_x.shape
        device = batch_x.device

        # 1) mask
        x_masked, mask = self._random_mask(batch_x)  # mask: [B,L,C] bool

        # 2) encode sensor -> [B,C,H]
        sensor_embeds = self._encode_sensor(x_masked)  # [B,C,H]

        # 3) build per-sample prompt (so mask summary is accurate per sample)
        prompts = []
        if self.include_mask_summary:
            for b in range(B):
                summary = self._mask_to_summary(mask[b])  # mask[b]: [L,C]
                prompts.append(self._build_prompt_text(mask_summary=summary))
        else:
            # cache a static prompt
            if self._prompt_text_cache is None:
                self._prompt_text_cache = self._build_prompt_text(mask_summary=None)
            prompts = [self._prompt_text_cache] * B

        # 4) tokenize prompt -> [B,T]
        input_ids, attn_mask = self._tokenize_prompts(prompts, device)

        # 5) inject embeddings -> [B,T,H]
        inputs_embeds = self._inject_sensor_embeds(input_ids, sensor_embeds)

        # 6) run LLaMA
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_hidden_states=self.output_hidden_states,
            use_cache=False,
        )

        # Meta: debugging / sanity
        meta = {
            "prompt_len": int(input_ids.shape[1]),
            "mask_rate_real": float(mask.float().mean().item()),
            "C": self.C,
            "L": int(L),
        }

        return out, mask, x_masked, meta
