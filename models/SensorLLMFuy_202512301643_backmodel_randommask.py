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


class Model(nn.Module):
    """
    Two-stage model (continuous MAE-style):
      - stage1/pretrain: patch-level masked reconstruction with MSE (continuous regression)
      - stage2/classify: HAR classification

    Input from dataloader:
      batch_x: [B, L, C]
      padding_mask: [B, L]  (optional, currently unused)
      labels: [B] or [B,1]  (optional)
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
        self.stage = int(getattr(args, "stage", 1))  # 1(pretrain) / 2(classify)

        # sequence + patch
        self.seq_len = int(getattr(args, "seq_len", 96))
        self.patch_len = int(getattr(args, "patch_len", 12))
        if self.seq_len % self.patch_len != 0:
            raise ValueError(f"seq_len={self.seq_len} must be divisible by patch_len={self.patch_len}")
        self.P = self.seq_len // self.patch_len  # number of patches

        # mask settings (patch-level)
        self.mask_rate = float(getattr(args, "mask_rate", 0.75))
        assert 0.0 <= self.mask_rate <= 1.0

        # prompt settings
        self.max_prompt_len = int(getattr(args, "max_prompt_len", 512))
        self.include_mask_summary = bool(getattr(args, "include_mask_summary", True))
        # max number of merged spans to show
        self.max_mask_spans_in_prompt = int(getattr(args, "max_mask_spans_in_prompt", 6))
        # show seconds or sample indices (seconds is more paper-friendly)
        self.mask_summary_in_seconds = bool(getattr(args, "mask_summary_in_seconds", True))
        # decimals for seconds formatting
        self.mask_summary_sec_ndigits = int(getattr(args, "mask_summary_sec_ndigits", 2))

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

        self.H = int(self.llm.config.hidden_size)

        # -----------------------
        # patch tokens: <TS_c_p> for each channel c and patch index p
        # K = C * P
        # -----------------------
        self.K = self.C * self.P
        self.ts_tokens: List[str] = [f"<TS_{c}_{p}>" for c in range(self.C) for p in range(self.P)]
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": self.ts_tokens})
        if added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))

        self.ts_token_ids: List[int] = [self.tokenizer.convert_tokens_to_ids(tok) for tok in self.ts_tokens]
        if any(t is None or t < 0 for t in self.ts_token_ids):
            raise RuntimeError("Failed to build <TS_c_p> token ids.")

        # -----------------------
        # sensor patch encoder -> [B, K, H]
        # -----------------------
        self.llm.sensor_patch_proj = nn.Linear(self.patch_len, self.H)
        self.llm.channel_id = nn.Embedding(self.C, self.H)
        self.llm.patch_pos = nn.Embedding(self.P, self.H)
        self.llm.mask_embed = nn.Parameter(torch.zeros(1, 1, self.H))  # used for masked patches

        # -----------------------
        # heads
        # pretrain: hidden -> patch_len (for each TS token)
        # classify: pooled hidden -> num_class
        # -----------------------
        self.llm.recon_head = nn.Linear(self.H, self.patch_len)
        # 在 __init__ 里（不要挂到 llm 里，挂到 self 里更安全）
        self.llm.pool_query = nn.Parameter(torch.zeros(1, 1, self.H))
        self.llm.pool_attn = nn.MultiheadAttention(self.H, num_heads=4, batch_first=True)

        self.llm.cls_head = nn.Linear(self.H, self.num_class)

        # prompt cache (when mask summary disabled)
        self._prompt_text_cache: Optional[str] = None

    # ============================================================
    # Patchify / Unpatchify
    # ============================================================
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        returns patches: [B, P, patch_len, C]
        """
        B, L, C = x.shape
        if L != self.seq_len:
            # truncate or left-pad with zeros to seq_len
            if L > self.seq_len:
                x = x[:, -self.seq_len:, :]
            else:
                pad = self.seq_len - L
                x = torch.cat([torch.zeros(B, pad, C, device=x.device, dtype=x.dtype), x], dim=1)
        patches = x.view(B, self.P, self.patch_len, C)
        return patches

    # ============================================================
    # Masking (patch-level)
    # ============================================================
    def _random_patch_mask(self, B: int, device) -> torch.Tensor:
        """
        returns patch_mask: [B, P] bool, True=masked
        """
        patch_mask = (torch.rand(B, self.P, device=device) < self.mask_rate)
        if self.P >= 2:
            all_masked = patch_mask.all(dim=1)
            none_masked = (~patch_mask).all(dim=1)
            if all_masked.any():
                patch_mask[all_masked, torch.randint(0, self.P, (all_masked.sum().item(),), device=device)] = False
            if none_masked.any():
                patch_mask[none_masked, torch.randint(0, self.P, (none_masked.sum().item(),), device=device)] = True
        return patch_mask

    # ============================================================
    # Encode sensor patches -> [B, K, H]
    # ============================================================
    def _encode_sensor_patches(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x: [B, L, C]
        patch_mask: [B, P] bool (True=masked)
        returns sensor_embeds: [B, K, H] aligned to self.ts_tokens order (c-major, then p)
        """
        patches = self._patchify(x)  # [B,P,T,C]
        patches_bpct = patches.permute(0, 1, 3, 2).contiguous()  # [B,P,C,T]
        B, P, C, T = patches_bpct.shape
        proj = self.llm.sensor_patch_proj(patches_bpct)  # [B,P,C,H]

        ch = torch.arange(self.C, device=x.device).view(1, 1, self.C).expand(B, P, self.C)
        pp = torch.arange(self.P, device=x.device).view(1, self.P, 1).expand(B, self.P, self.C)
        proj = proj + self.llm.channel_id(ch) + self.llm.patch_pos(pp)  # [B,P,C,H]
        # proj = proj +  self.patch_pos(pp)                # [B,P,C,H]

        if patch_mask is not None:
            m = patch_mask.view(B, P, 1, 1)
            proj = torch.where(m, self.llm.mask_embed.expand(B, P, self.C, self.H), proj)

        proj = proj.permute(0, 2, 1, 3).contiguous()  # [B,C,P,H]
        sensor_embeds = proj.view(B, self.K, self.H)  # [B,K,H]
        return sensor_embeds

    # ============================================================
    # Mask summary (paper-friendly)
    # ============================================================
    def _merge_masked_patch_spans(self, patch_mask_one: torch.Tensor) -> List[Tuple[int, int]]:
        """
        patch_mask_one: [P] bool
        returns merged spans as list of (p_start, p_end)
        """
        P = patch_mask_one.numel()
        spans: List[Tuple[int, int]] = []
        in_run = False
        s = 0
        for p in range(P):
            if patch_mask_one[p].item() and not in_run:
                in_run = True
                s = p
            if in_run and ((not patch_mask_one[p].item()) or (p == P - 1)):
                e = p if patch_mask_one[p].item() else (p - 1)
                spans.append((s, e))
                in_run = False
                if len(spans) >= self.max_mask_spans_in_prompt:
                    break
        return spans

    def _format_span(self, p_start: int, p_end: int) -> str:
        """
        Convert patch span to either sample index range or seconds range.
        """
        s_idx = p_start * self.patch_len
        e_idx = (p_end + 1) * self.patch_len - 1

        if self.mask_summary_in_seconds and self.sample_rate > 0:
            s_t = s_idx / float(self.sample_rate)
            e_t = e_idx / float(self.sample_rate)
            nd = max(0, int(self.mask_summary_sec_ndigits))
            # inclusive ranges
            return f"{s_t:.{nd}f}–{e_t:.{nd}f}s"
        else:
            return f"samples[{s_idx}-{e_idx}]"

    def _mask_to_summary_one(self, patch_mask_one: torch.Tensor) -> str:
        spans = self._merge_masked_patch_spans(patch_mask_one)
        if not spans:
            return "none"
        parts = [self._format_span(s, e) for (s, e) in spans]
        if len(spans) >= self.max_mask_spans_in_prompt:
            parts.append("...")
        return ", ".join(parts)

    # ============================================================
    # Prompt
    # ============================================================
    def _build_prompt_text(self, mask_summary: Optional[str] = None) -> str:
        # Ensure TS tokens appear as standalone tokens (space separated)
        lines = []
        lines.append("You are a sensor assistant.")
        lines.append(f"Dataset: {self.dataset_key}.")
        if self.sample_rate > 0:
            lines.append(f"Sampling rate: {self.sample_rate} Hz.")
        lines.append(f"Sequence length: {self.seq_len}. Patch length: {self.patch_len}. Patches: {self.P}.")
        lines.append(f"Channels: {self.C}. Each (channel, patch) uses a placeholder token.")
        if self.stage == 1:
            lines.append("Task: reconstruct masked segments in the time series (continuous values).")
        else:
            lines.append("Task: predict the activity class from sensor channels.")
        if mask_summary:
            lines.append(f"Masked segments: {mask_summary}")

        lines.append("Channel names: " + ", ".join(self.channel_names))
        lines.append("TS TOKENS: " + " ".join(self.ts_tokens))
        return "\n".join(lines)

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
    # Inject placeholder embeddings + gather TS token hiddens
    # ============================================================
    def _inject_sensor_embeds(self, input_ids: torch.Tensor, sensor_embeds: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B,T]
        sensor_embeds: [B,K,H] aligned with self.ts_token_ids
        returns inputs_embeds: [B,T,H]
        """
        B, T = input_ids.shape
        assert sensor_embeds.shape == (B, self.K, self.H)

        embed_layer = self.llm.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)  # [B,T,H]

        for k, ts_id in enumerate(self.ts_token_ids):
            pos = (input_ids == ts_id)  # [B,T]
            if not pos.any():
                continue
            b_idx, t_idx = pos.nonzero(as_tuple=True)
            inputs_embeds[b_idx, t_idx, :] = sensor_embeds[b_idx, k, :]
        return inputs_embeds

    def _gather_ts_hiddens(self, input_ids: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B,T]
        h: [B,T,H]
        returns ts_h: [B,K,H] hidden states at each TS token position
        """
        B, T = input_ids.shape
        ts_h = torch.zeros((B, self.K, self.H), device=h.device, dtype=h.dtype)
        for k, ts_id in enumerate(self.ts_token_ids):
            pos = (input_ids == ts_id)
            if not pos.any():
                continue
            b_idx, t_idx = pos.nonzero(as_tuple=True)
            ts_h[b_idx, k, :] = h[b_idx, t_idx, :]
        return ts_h

    # ============================================================
    # Forward
    # ============================================================
    def forward(self, batch_x, padding_mask=None, mode: Optional[str] = None, labels=None):
        """
        mode:
          - "pretrain" : return (loss_mse, patch_mask, meta)
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

        patch_mask = None
        if mode == "pretrain":
            patch_mask = self._random_patch_mask(B, x.device)  # [B,P] bool

        sensor_embeds = self._encode_sensor_patches(x, patch_mask=patch_mask)  # [B,K,H]

        # prompts
        if self.include_mask_summary and mode == "pretrain":
            prompts = [self._build_prompt_text(self._mask_to_summary_one(patch_mask[b])) for b in range(B)]
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
        ts_h = self._gather_ts_hiddens(input_ids, h)  # [B,K,H]

        if mode == "classify":
            # pooled = ts_h.mean(dim=1)                 # [B,H]
            # logits = self.llm.cls_head(pooled)            # [B,num_class]
            # forward classify 分支：
            q = self.llm.pool_query.expand(B, 1, self.H)  # [B,1,H]
            pooled, _ = self.llm.pool_attn(q, ts_h, ts_h)  # [B,1,H]
            pooled = pooled.squeeze(1)  # [B,H]
            logits = self.llm.cls_head(pooled)

            return logits

        # -------- pretrain (patch MAE continuous reconstruction) --------
        patch_vals = self.llm.recon_head(ts_h)  # [B,K,patch_len]
        patch_vals = patch_vals.view(B, self.C, self.P, self.patch_len)  # [B,C,P,T]
        pred_patches = patch_vals.permute(0, 2, 3, 1).contiguous()  # [B,P,T,C]
        gt_patches = self._patchify(x)  # [B,P,T,C]

        per_patch_mse = (pred_patches - gt_patches).pow(2).mean(dim=(2, 3))  # [B,P]
        loss_mse = per_patch_mse[patch_mask].mean()

        meta = {
            "prompt_len": int(input_ids.shape[1]),
            "mask_rate_real": float(patch_mask.float().mean().item()),
            "C": self.C,
            "L": int(self.seq_len),
            "P": int(self.P),
            "patch_len": int(self.patch_len),
            "mask_summary_example": self._mask_to_summary_one(patch_mask[0].detach().cpu()),
        }
        return loss_mse, patch_mask, meta

    # ============================================================
    # HF load helpers
    # ============================================================
    def load_hf_dir(self, hf_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_dir, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(hf_dir)
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.config.use_cache = False

        assert self.llm.get_input_embeddings().num_embeddings >= len(self.tokenizer), \
            f"Embedding too small: emb={self.llm.get_input_embeddings().num_embeddings}, vocab={len(self.tokenizer)}"


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
    model = Model(configs).to("cuda:0", dtype=torch.float16)
    x = torch.rand(32, 96, 15)
    c = model(x.to("cuda:0", dtype=torch.float16), None, None)
    d = 'end'