# models/SensorLLMFuy.py
import os
import json
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import yaml
except Exception:
    yaml = None


class Model(nn.Module):
    """
    Two-stage model:
      - stage1/pretrain: masked reconstruction with MSE (continuous regression)
      - stage2/classify: HAR classification

    Input from dataloader:
      batch_x: [B, L, C]
      padding_mask: [B, L]  (optional)
      label: [B, 1] or [B]
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # -----------------------
        # dataset cfg priority:
        #   1) args.ds_cfg injected by Exp (recommended)
        #   2) fallback load from yaml if user still passes path (optional)
        # -----------------------
        self.dataset_key = str(getattr(args, "dataset_key", getattr(args, "data", "mhealth"))).lower()
        self.ds_cfg: Dict[str, Any] = {}
        if hasattr(args, "ds_cfg") and isinstance(args.ds_cfg, dict):
            self.ds_cfg = args.ds_cfg
        else:
            # optional fallback (not recommended if Exp already injects)
            ts_yaml = getattr(args, "ts_backbone_yaml", None)
            if ts_yaml is not None:
                if yaml is None:
                    raise ImportError("pyyaml not installed but ts_backbone_yaml is set.")
                if not os.path.exists(ts_yaml):
                    raise FileNotFoundError(ts_yaml)
                with open(ts_yaml, "r", encoding="utf-8") as f:
                    cfg_all = yaml.safe_load(f)
                if self.dataset_key not in cfg_all:
                    raise KeyError(f"{self.dataset_key} not in {ts_yaml}")
                self.ds_cfg = cfg_all[self.dataset_key]

        # -----------------------
        # core sizes
        # -----------------------
        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 15)))
        self.num_class = int(self.ds_cfg.get("num_labels", getattr(args, "num_class", 12)))
        self.sample_rate = int(self.ds_cfg.get("sample_rate", getattr(args, "sample_rate", 0)))

        self.channel_names: List[str] = self.ds_cfg.get("channel_names", None)
        if self.channel_names is None:
            self.channel_names = [f"ch{i}" for i in range(self.C)]
        else:
            if len(self.channel_names) != self.C:
                raise ValueError(f"channel_names length {len(self.channel_names)} != C {self.C}")

        # stage control
        # args.stage: 1 (pretrain) / 2 (classify). you can still override via forward(mode=...)
        self.stage = args.stage

        # mask settings
        self.mask_rate = float(getattr(args, "mask_rate", 0.3))
        assert 0.0 <= self.mask_rate <= 1.0

        # prompt settings
        self.max_prompt_len = int(getattr(args, "max_prompt_len", 512))
        self.include_mask_summary = bool(getattr(args, "include_mask_summary", False))
        self.max_mask_spans_in_prompt = int(getattr(args, "max_mask_spans_in_prompt", 8))

        # LLaMA path/name
        self.llama_name = getattr(args, "llama_name", None)
        if self.llama_name is None:
            raise ValueError("args.llama_name is required (path or HF id).")

        # -----------------------
        # tokenizer + llm
        # -----------------------
        self.tokenizer = AutoTokenizer.from_pretrained(self.llama_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_dtype = getattr(args, "llm_dtype", "float16")
        torch_dtype = getattr(torch, llm_dtype, torch.float16)

        self.llm = AutoModelForCausalLM.from_pretrained(self.llama_name, torch_dtype=torch_dtype)
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.config.use_cache = False

        H = int(self.llm.config.hidden_size)

        # -----------------------
        # ts placeholder tokens
        # -----------------------
        self.ts_tokens = [f"<TS_{i}>" for i in range(self.C)]
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": self.ts_tokens})
        if added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))

        self.ts_token_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in self.ts_tokens]
        if any(t is None or t < 0 for t in self.ts_token_ids):
            raise RuntimeError("Failed to build <TS_i> token ids.")

        # -----------------------
        # sensor encoder -> [B,C,H] (simple baseline)
        # -----------------------
        self.llm.sensor_proj = nn.Linear(self.args.seq_len, H)
        self.llm.channel_id = nn.Embedding(self.C, H)

        # -----------------------
        # heads
        # pretrain: token hidden -> C (then broadcast to L)
        # classify: pooled hidden -> num_class
        # -----------------------
        self.llm.recon_head = nn.Linear(H, self.C)
        self.llm.cls_head = nn.Linear(H, self.num_class)

        # IMPORTANT: attach to llm for HF save_pretrained()
        # self.llm.sensor_proj = self.sensor_proj
        # self.llm.channel_id = self.channel_id
        # self.llm.recon_head = self.recon_head
        # self.llm.cls_head = self.cls_head

        # prompt cache (when mask summary disabled)
        self._prompt_text_cache: Optional[str] = None

    # ============================================================
    # Masking
    # ============================================================
    def _random_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,L,C]
        B, L, C = x.shape
        assert C == self.C
        mask = (torch.rand(B, L, C, device=x.device) < self.mask_rate)
        x_masked = x.masked_fill(mask, 0.0)
        return x_masked, mask

    # ============================================================
    # Encode sensor -> [B,C,H]
    # ============================================================
    def _encode_sensor(self, x_masked: torch.Tensor) -> torch.Tensor:
        # mean pool time: [B,C]
        pooled = x_masked.mean(dim=1).unsqueeze(-1)    # [B,C,1]
        # emb = self.llm.sensor_proj(x_masked.permute(0,2,1))                 # [B,C,H]
        emb = self.llm.sensor_proj(x_masked.permute(0,2,1))                 # [B,C,H]
        ch = torch.arange(self.C, device=x_masked.device).unsqueeze(0).expand(x_masked.size(0), self.C)
        emb = emb + self.llm.channel_id(ch)
        return emb

    # ============================================================
    # Prompt
    # ============================================================
    def _build_prompt_text(self, mask_summary: Optional[str] = None) -> str:
        lines = []
        lines.append("You are a sensor assistant.")
        lines.append(f"Dataset: {self.dataset_key}.")
        if self.sample_rate > 0:
            lines.append(f"Sampling rate: {self.sample_rate} Hz.")
        lines.append(f"Channels: {self.C}. Each channel is represented by a placeholder token.")
        if self.stage == 1:
            lines.append("Task: reconstruct masked values in the time series.")
        else:
            lines.append("Task: predict the activity class from sensor channels.")
        if mask_summary:
            lines.append(f"Masked summary: {mask_summary}")
        lines.append("Channel placeholders:")
        for i, name in enumerate(self.channel_names):
            lines.append(f"{name}: <TS_{i}>")
        return "\n".join(lines)

    def _mask_to_summary_one(self, mask_lc: torch.Tensor) -> str:
        # mask_lc: [L,C]
        L, C = mask_lc.shape
        spans = []
        for c in range(C):
            m = mask_lc[:, c]
            in_run = False
            s = 0
            for t in range(L):
                if m[t].item() and not in_run:
                    in_run = True
                    s = t
                if in_run and ((not m[t].item()) or (t == L - 1)):
                    e = t if m[t].item() else (t - 1)
                    spans.append((c, s, e))
                    in_run = False
                    if len(spans) >= self.max_mask_spans_in_prompt:
                        break
            if len(spans) >= self.max_mask_spans_in_prompt:
                break
        if not spans:
            return "none"
        parts = []
        for c, s, e in spans:
            cname = self.channel_names[c] if c < len(self.channel_names) else f"ch{c}"
            parts.append(f"{cname}[{s}-{e}]" if s != e else f"{cname}[{s}]")
        return ", ".join(parts)

    def _tokenize_prompts(self, prompts: List[str], device) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_len,
            return_tensors="pt",
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    # ============================================================
    # Inject placeholder embeddings
    # ============================================================
    def _inject_sensor_embeds(self, input_ids: torch.Tensor, sensor_embeds: torch.Tensor) -> torch.Tensor:
        # input_ids: [B,T], sensor_embeds: [B,C,H]
        B, T = input_ids.shape
        B2, C, H = sensor_embeds.shape
        assert B2 == B and C == self.C
        embed_layer = self.llm.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)  # [B,T,H]

        for i, ts_id in enumerate(self.ts_token_ids):
            pos = (input_ids == ts_id)  # [B,T]
            if not pos.any():
                continue
            for b in range(B):
                idxs = torch.nonzero(pos[b], as_tuple=False).flatten()
                if idxs.numel() == 0:
                    continue
                inputs_embeds[b, idxs, :] = sensor_embeds[b, i, :].unsqueeze(0).expand(idxs.numel(), H)
        return inputs_embeds

    # ============================================================
    # Forward
    # ============================================================
    def forward(self, batch_x, padding_mask=None, mode: Optional[str] = None, labels=None):
        """
        mode:
          - "pretrain" : return (loss_mse, mask, x_masked, meta)
          - "classify" : return logits [B,num_class]
        """
        if mode is None:
            mode = "pretrain" if self.stage == 1 else "classify"

        if not torch.is_tensor(batch_x):
            batch_x = torch.as_tensor(batch_x)
        x = batch_x
        if x.dim() != 3:
            raise ValueError(f"batch_x must be [B,L,C], got {tuple(x.shape)}")
        B, L, C = x.shape
        if C != self.C:
            raise ValueError(f"Input C={C} != model C={self.C}. Check ds_cfg/enc_in.")

        # stage1 uses random mask
        if mode == "pretrain":
            x_masked, mask = self._random_mask(x)             # [B,L,C], bool [B,L,C]
        else:
            x_masked = x
            mask = None

        sensor_embeds = self._encode_sensor(x_masked)         # [B,C,H]

        # prompts
        if self.include_mask_summary and mode == "pretrain":
            prompts = [self._build_prompt_text(self._mask_to_summary_one(mask[b])) for b in range(B)]
        else:
            if self._prompt_text_cache is None:
                self._prompt_text_cache = self._build_prompt_text(None)
            prompts = [self._prompt_text_cache] * B

        input_ids, attn_mask = self._tokenize_prompts(prompts, x.device)
        inputs_embeds = self._inject_sensor_embeds(input_ids, sensor_embeds)

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        h = out.hidden_states[-1]  # [B,T,H]

        if mode == "classify":
            pooled = h[:, -1, :]                       # [B,H]
            logits = self.llm.cls_head(pooled)         # [B,num_class]
            return logits

        # -------- pretrain (MSE continuous regression) --------
        # token-level -> C
        x_hat_tokens = self.llm.recon_head(h)          # [B,T,C]

        # minimal runnable alignment:
        # global prediction per channel then broadcast to L
        x_hat_global = x_hat_tokens.mean(dim=1)        # [B,C]
        x_hat = x_hat_global.unsqueeze(1).expand(-1, L, -1)  # [B,L,C]

        diff = x_hat - x
        loss_mse = (diff[mask] ** 2).mean() if mask.any() else (diff ** 2).mean()

        meta = {
            "prompt_len": int(input_ids.shape[1]),
            "mask_rate_real": float(mask.float().mean().item()),
            "C": self.C,
            "L": int(L),
        }
        return loss_mse, mask, x_masked, meta

    # ============================================================
    # HF load helpers
    # ============================================================
    def load_hf_dir(self, hf_dir: str):
        # load llm + tokenizer from stage1 directory
        self.tokenizer = AutoTokenizer.from_pretrained(hf_dir, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(hf_dir)
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.config.use_cache = False

        # strict safety
        assert self.llm.get_input_embeddings().num_embeddings >= len(self.tokenizer), \
            f"Embedding too small: emb={self.llm.get_input_embeddings().num_embeddings}, vocab={len(self.tokenizer)}"
