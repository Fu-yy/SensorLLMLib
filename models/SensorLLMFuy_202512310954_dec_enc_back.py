# models/SensorLLMFuy.py
import os
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.FackLLM import FakeLLM

try:
    import yaml
except Exception:
    yaml = None


# models/SensorLLMResampler.py
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import yaml
except Exception:
    yaml = None


# ----------------------------
# 1) Resampler (Perceiver/Q-Former style)
# ----------------------------
class SensorResampler(nn.Module):
    """
    Query-based resampler:
      Z: [B, K, H] -> R: [B, M, H]
    """
    def __init__(self, hidden_size: int, num_latents: int = 16, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.M = int(num_latents)
        self.H = int(hidden_size)
        self.query = nn.Parameter(torch.zeros(1, self.M, self.H))
        nn.init.normal_(self.query, std=0.02)

        self.attn = nn.MultiheadAttention(self.H, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ln_q = nn.LayerNorm(self.H)
        self.ln_kv = nn.LayerNorm(self.H)

        # small FFN to increase capacity (still light)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.H),
            nn.Linear(self.H, 4 * self.H),
            nn.GELU(),
            nn.Linear(4 * self.H, self.H),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, K, H]
        """
        B, K, H = z.shape
        q = self.query.expand(B, -1, -1)  # [B,M,H]
        q = self.ln_q(q)
        z = self.ln_kv(z)

        r, _ = self.attn(q, z, z, need_weights=False)     # [B,M,H]
        r = r + self.ffn(r)
        return r


# ----------------------------
# 2) Continuous position encoding that generalizes to any P
# ----------------------------
class ContinuousPosMLP(nn.Module):
    """
    Given normalized position t in [0,1], produce pos embedding [H].
    This avoids nn.Embedding(P,H) which breaks when P changes.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        H = int(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(1, H),
            nn.GELU(),
            nn.Linear(H, H),
        )

    def forward(self, pos01: torch.Tensor) -> torch.Tensor:
        """
        pos01: [P] or [B,P] float in [0,1]
        return: [..., H]
        """
        x = pos01.unsqueeze(-1)  # [...,1]
        return self.net(x)


# ============================================================
# Route A: Frozen LLM + Resampler bottleneck (two-stage)
# ============================================================
class Model(nn.Module):
    """
    Route A (recommended main):
      Stage1: masked patch reconstruction (MSE on masked patches)
      Stage2: classification (CE)
    Both stages go through frozen LLM.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # -------- dataset cfg load (same as your original) --------
        self.dataset_key = str(getattr(args, "dataset_key", getattr(args, "data", "mhealth"))).lower()
        self.ds_cfg: Dict[str, Any] = {}
        if hasattr(args, "ds_cfg") and isinstance(args.ds_cfg, dict):
            self.ds_cfg = args.ds_cfg
        else:
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

        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 15)))
        self.num_class = int(self.ds_cfg.get("num_labels", getattr(args, "num_class", 12)))
        self.sample_rate = int(self.ds_cfg.get("sample_rate", getattr(args, "sample_rate", 0)))
        self.channel_names: List[str] = self.ds_cfg.get("channel_names", [f"ch{i}" for i in range(self.C)])

        # stage control
        self.stage = int(getattr(args, "stage", 1))  # 1 pretrain / 2 classify

        # seq/patch can vary across runs; model should not bake in P
        self.seq_len = int(getattr(args, "seq_len", 96))
        self.patch_len = int(getattr(args, "patch_len", 12))
        if self.seq_len % self.patch_len != 0:
            raise ValueError(f"seq_len={self.seq_len} must be divisible by patch_len={self.patch_len}")

        self.mask_rate = float(getattr(args, "mask_rate", 0.75))
        assert 0.0 <= self.mask_rate <= 1.0

        # prompt
        self.max_prompt_len = int(getattr(args, "max_prompt_len", 256))
        self.llama_name = getattr(args, "llama_name", None)
        if self.llama_name is None:
            raise ValueError("args.llama_name is required (path or HF id).")

        # -------- tokenizer + frozen llm --------
        self.tokenizer = AutoTokenizer.from_pretrained(self.llama_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # llm_dtype = getattr(args, "llm_dtype", "float16")
        # torch_dtype = getattr(torch, llm_dtype, torch.float16)
        llm_dtype = getattr(args, "llm_dtype", "float16")
        torch_dtype = getattr(torch, llm_dtype, torch.float16)
        self.llm = AutoModelForCausalLM.from_pretrained(self.llama_name, torch_dtype=torch_dtype)
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.config.use_cache = False
        self.H = int(self.llm.config.hidden_size)

        # Freeze LLM (Exp will also do this, but safe here)
        if bool(getattr(args, "freeze_llm", True)):
            self.llm.requires_grad_(False)

        # -------- sensor patch encoder (learnable, small) --------
        # patchify: [B,P,patch_len,C] -> per-token embed H
        self.sensor_patch_proj = nn.Linear(self.patch_len, self.H)
        self.channel_id = nn.Embedding(self.C, self.H)

        # position: continuous pos MLP so P can change
        self.pos_mlp = ContinuousPosMLP(self.H)

        self.mask_embed = nn.Parameter(torch.zeros(1, 1, self.H))
        nn.init.normal_(self.mask_embed, std=0.02)

        # -------- resampler bottleneck (fixed M) --------
        self.num_latents = int(getattr(args, "num_latents", 16))  # M
        self.resampler = SensorResampler(self.H, num_latents=self.num_latents, num_heads=4, dropout=0.0)

        # -------- heads / decoders --------
        # Stage1 decoder: from LLM-processed latents -> predict all patches (P*C*patch_len) for this sample
        # We decode into [B,P,C,patch_len], then compute MSE on masked patches.
        self.recon_decoder = nn.Sequential(
            nn.LayerNorm(self.H),
            nn.Linear(self.H, self.H),
            nn.GELU(),
            nn.Linear(self.H, self.H),
        )
        # A light cross-attn decoder: queries = (patch,channel) tokens, keys/values = latent tokens
        self.dec_num_heads = 4
        self.dec_attn = nn.MultiheadAttention(self.H, num_heads=self.dec_num_heads, batch_first=True)

        self.recon_head = nn.Linear(self.H, self.patch_len)

        # Stage2 pooling + cls
        self.pool_query = nn.Parameter(torch.zeros(1, 1, self.H))
        nn.init.normal_(self.pool_query, std=0.02)
        self.pool_attn = nn.MultiheadAttention(self.H, num_heads=4, batch_first=True)
        self.cls_head = nn.Linear(self.H, self.num_class)

        # prompt cache (keep simple; you can enrich)
        self._prompt_cache = None

    # -------------------------
    # patchify (handles varying L at runtime)
    # -------------------------
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,L,C]
        returns [B,P,patch_len,C] with zero-left-pad or truncate to self.seq_len
        """
        B, L, C = x.shape
        if L != self.seq_len:
            if L > self.seq_len:
                x = x[:, -self.seq_len:, :]
            else:
                pad = self.seq_len - L
                x = torch.cat([torch.zeros(B, pad, C, device=x.device, dtype=x.dtype), x], dim=1)
        P = self.seq_len // self.patch_len
        return x.view(B, P, self.patch_len, C)

    def _random_patch_mask(self, B: int, P: int, device) -> torch.Tensor:
        patch_mask = (torch.rand(B, P, device=device) < self.mask_rate)
        if P >= 2:
            all_masked = patch_mask.all(dim=1)
            none_masked = (~patch_mask).all(dim=1)
            if all_masked.any():
                patch_mask[all_masked, torch.randint(0, P, (all_masked.sum().item(),), device=device)] = False
            if none_masked.any():
                patch_mask[none_masked, torch.randint(0, P, (none_masked.sum().item(),), device=device)] = True
        return patch_mask

    # -------------------------
    # Encode sensor -> Z tokens: [B, K, H] where K=P*C
    # -------------------------
    def _encode_sensor_tokens(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, int]:
        patches = self._patchify(x)                 # [B,P,T,C]
        B, P, T, C = patches.shape

        # [B,P,C,T]
        bpct = patches.permute(0, 1, 3, 2).contiguous()
        z = self.sensor_patch_proj(bpct)           # [B,P,C,H]

        # channel embed
        ch = torch.arange(self.C, device=x.device).view(1, 1, self.C).expand(B, P, self.C)
        z = z + self.channel_id(ch)                # [B,P,C,H]

        # continuous pos embed
        pos = torch.linspace(0.0, 1.0, steps=P, device=x.device, dtype=z.dtype)  # [P]
        pos_e = self.pos_mlp(pos).view(1, P, 1, self.H)                          # [1,P,1,H]
        z = z + pos_e

        # mask patches (mask applied to all channels at that patch)
        if patch_mask is not None:
            m = patch_mask.view(B, P, 1, 1)
            z = torch.where(m, self.mask_embed.view(1, 1, 1, self.H).expand(B, P, self.C, self.H), z)

        # flatten to K=P*C in c-major? keep consistent: [B,P,C,H] -> [B,K,H]
        z = z.view(B, P * self.C, self.H)
        return z, P

    # -------------------------
    # Prompt text -> embeds
    # -------------------------
    def _build_prompt(self) -> str:
        # keep prompt short and stable (length-generalization paper likes this)
        return f"You are a sensor assistant. Dataset: {self.dataset_key}. Task: analyze sensor tokens."

    def _get_text_embeds(self, B: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._prompt_cache is None:
            self._prompt_cache = self._build_prompt()
        prompts = [self._prompt_cache] * B
        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [B,T,H]
        return text_embeds, attn_mask

    # -------------------------
    # LLM forward with appended sensor tokens
    # -------------------------
    def _llm_encode_latents(self, latents: torch.Tensor, text_embeds: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        """
        latents: [B,M,H]
        returns processed_latents: [B,M,H] from last hidden layer (positions corresponding to appended latents)
        """

        # print("llm emb dtype:", self.llm.get_input_embeddings().weight.dtype)
        # print("text_embeds dtype:", text_embeds.dtype)
        # print("latents dtype:", latents.dtype)

        B, T, H = text_embeds.shape
        M = latents.shape[1]
        embed_layer = self.llm.get_input_embeddings()
        target_dtype = embed_layer.weight.dtype  # 通常是 torch.float16
        target_device = embed_layer.weight.device

        # 保证都在同 device + 同 dtype
        text_embeds = text_embeds.to(device=target_device, dtype=target_dtype)
        latents = latents.to(device=target_device, dtype=target_dtype)
        text_mask = text_mask.to(device=target_device)
        inputs_embeds = torch.cat([text_embeds, latents], dim=1)               # [B,T+M,H]
        attn_mask = torch.cat([text_mask, torch.ones(B, M, device=text_mask.device, dtype=text_mask.dtype)], dim=1)

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        h = out.hidden_states[-1]                                             # [B,T+M,H]
        return h[:, -M:, :]                                                   # [B,M,H]

    # -------------------------
    # Decode latents -> patch predictions [B,P,C,patch_len]
    # -------------------------
    def _decode_patches_from_latents(self, latents_proc: torch.Tensor, P: int) -> torch.Tensor:
        """
        latents_proc: [B,M,H]  (after LLM)
        We create queries for each (p,c): Q_pc = channel_embed(c) + pos_embed(p)
        Then cross-attend to latents, then recon_head -> patch_len
        return pred: [B,P,C,patch_len]
        """
        B, M, H = latents_proc.shape

        # build queries [B, P*C, H]
        pos = torch.linspace(0.0, 1.0, steps=P, device=latents_proc.device, dtype=latents_proc.dtype)  # [P]
        pos_e = self.pos_mlp(pos).view(1, P, 1, H)                                                     # [1,P,1,H]
        ch = torch.arange(self.C, device=latents_proc.device).view(1, 1, self.C).expand(1, P, self.C)  # [1,P,C]
        ch_e = self.channel_id(ch).view(1, P, self.C, H)                                                # [1,P,C,H]

        q = (pos_e + ch_e).view(1, P * self.C, H).expand(B, -1, -1)                                     # [B,PC,H]
        q = self.recon_decoder(q)

        # cross-attn: queries=PC, keys=latents
        z, _ = self.dec_attn(q, latents_proc, latents_proc, need_weights=False)                         # [B,PC,H]
        patch_vals = self.recon_head(z)                                                                 # [B,PC,patch_len]
        patch_vals = patch_vals.view(B, P, self.C, self.patch_len)                                      # [B,P,C,T]
        return patch_vals

    # ============================================================
    # Forward
    # ============================================================
    def forward(self, batch_x, padding_mask=None, mode: Optional[str] = None, labels=None):
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

        # encode sensor tokens
        patch_mask = None
        # compute P for current seq_len/patch_len
        P = self.seq_len // self.patch_len

        if mode == "pretrain":
            patch_mask = self._random_patch_mask(B, P, x.device)             # [B,P] bool

        z, P_runtime = self._encode_sensor_tokens(x, patch_mask)             # z:[B,PC,H]
        assert P_runtime == P

        # resample to fixed M
        latents = self.resampler(z)                                          # [B,M,H]

        # run frozen LLM (Route A key)
        text_embeds, text_mask = self._get_text_embeds(B, x.device)
        latents_proc = self._llm_encode_latents(latents, text_embeds, text_mask)  # [B,M,H]

        if mode == "classify":
            q = self.pool_query.expand(B, 1, self.H)
            pooled, _ = self.pool_attn(q, latents_proc, latents_proc, need_weights=False)
            pooled = pooled.squeeze(1)
            logits = self.cls_head(pooled)
            return logits

        # -------- Stage1: reconstruct masked patches --------
        # pred: [B,P,C,T]
        pred_pcT = self._decode_patches_from_latents(latents_proc, P)        # [B,P,C,patch_len]
        # gt: [B,P,T,C] -> [B,P,C,T]
        gt = self._patchify(x).permute(0, 1, 3, 2).contiguous()              # [B,P,C,T]

        # MSE per patch (average over C and T)
        per_patch_mse = (pred_pcT - gt).pow(2).mean(dim=(2, 3))              # [B,P]
        loss = per_patch_mse[patch_mask].mean()

        meta = {
            "P": int(P),
            "C": int(self.C),
            "patch_len": int(self.patch_len),
            "num_latents": int(self.num_latents),
            "mask_rate_real": float(patch_mask.float().mean().item()),
        }
        return loss, patch_mask, meta

    # -------------------------
    # Save/load wrapper (interface) weights
    # -------------------------
    def save_wrapper(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        torch.save(self.state_dict(), path)

    def load_wrapper(self, path: str, map_location="cpu"):
        sd = torch.load(path, map_location=map_location)
        self.load_state_dict(sd, strict=True)


# ============================================================
# Route B: No LLM (two-stage) - for ablation / resource saving
# ============================================================
class ModelB_NoLLM_Resampler(nn.Module):
    """
    Route B (ablation):
      Stage1: masked reconstruction from bottleneck tokens (no LLM)
      Stage2: classification from bottleneck tokens (no LLM)
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dataset_key = str(getattr(args, "dataset_key", getattr(args, "data", "mhealth"))).lower()
        self.ds_cfg: Dict[str, Any] = {}
        if hasattr(args, "ds_cfg") and isinstance(args.ds_cfg, dict):
            self.ds_cfg = args.ds_cfg

        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 15)))
        self.num_class = int(self.ds_cfg.get("num_labels", getattr(args, "num_class", 12)))
        self.stage = int(getattr(args, "stage", 1))

        self.seq_len = int(getattr(args, "seq_len", 96))
        self.patch_len = int(getattr(args, "patch_len", 12))
        if self.seq_len % self.patch_len != 0:
            raise ValueError(f"seq_len={self.seq_len} must be divisible by patch_len={self.patch_len}")
        self.mask_rate = float(getattr(args, "mask_rate", 0.75))

        # choose a hidden size (since no LLM, you must specify)
        self.H = int(getattr(args, "bottleneck_hidden", 512))

        # sensor encoder
        self.sensor_patch_proj = nn.Linear(self.patch_len, self.H)
        self.channel_id = nn.Embedding(self.C, self.H)
        self.pos_mlp = ContinuousPosMLP(self.H)
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, self.H))
        nn.init.normal_(self.mask_embed, std=0.02)

        # resampler
        self.num_latents = int(getattr(args, "num_latents", 16))
        self.resampler = SensorResampler(self.H, num_latents=self.num_latents, num_heads=4, dropout=0.0)

        # Stage1 decoder (same idea, but keys/values are latents directly)
        self.recon_decoder = nn.Sequential(
            nn.LayerNorm(self.H),
            nn.Linear(self.H, self.H),
            nn.GELU(),
            nn.Linear(self.H, self.H),
        )
        self.dec_attn = nn.MultiheadAttention(self.H, num_heads=4, batch_first=True)
        self.recon_head = nn.Linear(self.H, self.patch_len)

        # Stage2 pooling + cls
        self.pool_query = nn.Parameter(torch.zeros(1, 1, self.H))
        nn.init.normal_(self.pool_query, std=0.02)
        self.pool_attn = nn.MultiheadAttention(self.H, num_heads=4, batch_first=True)
        self.cls_head = nn.Linear(self.H, self.num_class)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        if L != self.seq_len:
            if L > self.seq_len:
                x = x[:, -self.seq_len:, :]
            else:
                pad = self.seq_len - L
                x = torch.cat([torch.zeros(B, pad, C, device=x.device, dtype=x.dtype), x], dim=1)
        P = self.seq_len // self.patch_len
        return x.view(B, P, self.patch_len, C)

    def _random_patch_mask(self, B: int, P: int, device) -> torch.Tensor:
        patch_mask = (torch.rand(B, P, device=device) < self.mask_rate)
        if P >= 2:
            all_masked = patch_mask.all(dim=1)
            none_masked = (~patch_mask).all(dim=1)
            if all_masked.any():
                patch_mask[all_masked, torch.randint(0, P, (all_masked.sum().item(),), device=device)] = False
            if none_masked.any():
                patch_mask[none_masked, torch.randint(0, P, (none_masked.sum().item(),), device=device)] = True
        return patch_mask

    def _encode_sensor_tokens(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, int]:
        patches = self._patchify(x)                 # [B,P,T,C]
        B, P, T, C = patches.shape
        bpct = patches.permute(0, 1, 3, 2).contiguous()   # [B,P,C,T]
        z = self.sensor_patch_proj(bpct)                 # [B,P,C,H]

        ch = torch.arange(self.C, device=x.device).view(1, 1, self.C).expand(B, P, self.C)
        z = z + self.channel_id(ch)

        pos = torch.linspace(0.0, 1.0, steps=P, device=x.device, dtype=z.dtype)
        pos_e = self.pos_mlp(pos).view(1, P, 1, self.H)
        z = z + pos_e

        if patch_mask is not None:
            m = patch_mask.view(B, P, 1, 1)
            z = torch.where(m, self.mask_embed.view(1, 1, 1, self.H).expand(B, P, self.C, self.H), z)

        z = z.view(B, P * self.C, self.H)
        return z, P

    def _decode_patches_from_latents(self, latents: torch.Tensor, P: int) -> torch.Tensor:
        B, M, H = latents.shape

        pos = torch.linspace(0.0, 1.0, steps=P, device=latents.device, dtype=latents.dtype)
        pos_e = self.pos_mlp(pos).view(1, P, 1, H)
        ch = torch.arange(self.C, device=latents.device).view(1, 1, self.C).expand(1, P, self.C)
        ch_e = self.channel_id(ch).view(1, P, self.C, H)

        q = (pos_e + ch_e).view(1, P * self.C, H).expand(B, -1, -1)
        q = self.recon_decoder(q)
        z, _ = self.dec_attn(q, latents, latents, need_weights=False)
        patch_vals = self.recon_head(z).view(B, P, self.C, self.patch_len)
        return patch_vals

    def forward(self, batch_x, padding_mask=None, mode: Optional[str] = None, labels=None):
        if mode is None:
            mode = "pretrain" if self.stage == 1 else "classify"

        if not torch.is_tensor(batch_x):
            batch_x = torch.as_tensor(batch_x)
        x = batch_x
        B, L, C = x.shape
        if C != self.C:
            raise ValueError(f"Input C={C} != model C={self.C}.")

        P = self.seq_len // self.patch_len
        patch_mask = None
        if mode == "pretrain":
            patch_mask = self._random_patch_mask(B, P, x.device)

        z, _ = self._encode_sensor_tokens(x, patch_mask)
        latents = self.resampler(z)  # [B,M,H]

        if mode == "classify":
            q = self.pool_query.expand(B, 1, self.H)
            pooled, _ = self.pool_attn(q, latents, latents, need_weights=False)
            pooled = pooled.squeeze(1)
            return self.cls_head(pooled)

        pred_pcT = self._decode_patches_from_latents(latents, P)                     # [B,P,C,T]
        gt = self._patchify(x).permute(0, 1, 3, 2).contiguous()                      # [B,P,C,T]
        per_patch_mse = (pred_pcT - gt).pow(2).mean(dim=(2, 3))                      # [B,P]
        loss = per_patch_mse[patch_mask].mean()
        meta = {"P": int(P), "C": int(self.C), "patch_len": int(self.patch_len), "num_latents": int(self.num_latents)}
        return loss, patch_mask, meta

    def save_wrapper(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        torch.save(self.state_dict(), path)

    def load_wrapper(self, path: str, map_location="cpu"):
        sd = torch.load(path, map_location=map_location)
        self.load_state_dict(sd, strict=True)





def get_configs():
    import random
    import numpy as np
    import argparse
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # datasets loader
    parser.add_argument('--datasets', type=str, required=False, default='ETTh1', help='dataset type')
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./datasets/ETT/', help='root path of the datasets file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='datasets file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output datasets', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='datasets loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input datasets')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    # SensorLLMFuy
    parser.add_argument('--ts_backbone_yaml', type=str, default='ts_backbone.yaml', help='ts_backbone_yaml')
    parser.add_argument('--log_dir', type=str, default='./logs', help='log_dir')
    parser.add_argument('--dataset_key', type=str, default='mhealth', help='dataset_key')
    parser.add_argument('--stage', type=int, default=1, help='stage')
    parser.add_argument('--llama_name', type=str, default=r"D:\fuy\MyCode\SensorLLM\Llama-3.2-1B", help='stage')
    parser.add_argument('--two_stage', type=int, default=1, help='two_stage')
    parser.add_argument('--freeze_llm', type=int, default=1, help='freeze_llm')
    parser.add_argument('--trainable_modules', type=str, default="sensor_proj,channel_id,recon_head,cls_head",
                        help='trainable_modules')
    parser.add_argument('--run_id', type=str, default="202501220_0956", help='trainable_modules')
    parser.add_argument('--debug_fake_llm', type=bool, default=False, help='debug_fake_llm')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    configs = get_configs()
    configs.ts_backbone_yaml = r"D:\fuy\MyCode\SensorLLMLib\configs\ts_backbone.yaml"
    configs.debug_fake_llm = True
    model = Model(configs).to("cuda:0")
    x = torch.rand(32,96,15)
    c = model(x.to("cuda:0"),None,None)
    d = 'end'