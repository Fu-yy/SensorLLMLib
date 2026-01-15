# models/SensorLLMResampler.py  (Route A)
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
# 禁用 fused attention，避免 _efficient_attention_backward invalid argument
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
try:
    import yaml
except Exception:
    yaml = None

# models/SensorLLM_RouteA_Refined.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# 禁用 fused attention 以兼容旧硬件
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

try:
    import yaml
except Exception:
    yaml = None

PRETRAIN_trainable_modules="patch_embed,resampler,proj_head"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"

# ============================================================
# 1. Scientific Components (Tokenizer & Deep Resampler)
# ============================================================

class PatchEmbedding(nn.Module):
    """
    [Scientific Fix 1]: Channel Mixing Tokenizer.
    将 (Patch_len * Channels) 视为一个整体 Token，保留多轴协同信息。
    Input: [B, Seq_Len, C] -> Output: [B, Num_Patches, Embed_Dim]
    """

    def __init__(self, seq_len: int, patch_len: int, in_channels: int, embed_dim: int):
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by patch_len={patch_len}")

        self.num_patches = seq_len // patch_len
        self.patch_len = patch_len

        # 核心差异：输入维度是 patch_len * in_channels
        self.proj = nn.Linear(patch_len * in_channels, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # 绝对位置编码 (Learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        B, L, C = x.shape
        # Flatten Patch & Channel: [B, Num_Patches, Patch_Len * C]
        x = x.view(B, self.num_patches, self.patch_len * C)
        x = self.proj(x)
        x = self.norm(x)
        x = x + self.pos_embed
        return x


class ResamplerLayer(nn.Module):
    """
    [Scientific Fix 2]: Standard Resampler Block
    Self-Attn (Latent mixing) -> Cross-Attn (Latent query Sensor) -> FFN
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln_latents = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)

        self.ln_cross_q = nn.LayerNorm(hidden_size)
        self.ln_cross_kv = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)

        self.ln_ffn = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, latents: torch.Tensor, sensor_embeds: torch.Tensor) -> torch.Tensor:
        # 1. Self-Attention
        q_sa = self.ln_latents(latents)
        latents_out, _ = self.self_attn(q_sa, q_sa, q_sa)
        latents = latents + latents_out

        # 2. Cross-Attention
        q_ca = self.ln_cross_q(latents)
        k_ca = v_ca = self.ln_cross_kv(sensor_embeds)
        cross_out, _ = self.cross_attn(query=q_ca, key=k_ca, value=v_ca)
        latents = latents + cross_out

        # 3. FFN
        ffn_out = self.ffn(self.ln_ffn(latents))
        latents = latents + ffn_out
        return latents


class DeepResampler(nn.Module):
    """
    Deep Information Bottleneck
    """

    def __init__(self, hidden_size: int, num_latents: int, depth: int = 4, num_heads: int = 8):
        super().__init__()
        self.num_latents = num_latents

        # Learnable Queries
        self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_size))
        nn.init.trunc_normal_(self.latents, std=0.02)

        self.layers = nn.ModuleList([
            ResamplerLayer(hidden_size, num_heads) for _ in range(depth)
        ])

        self.final_ln = nn.LayerNorm(hidden_size)

    def forward(self, sensor_embeds: torch.Tensor) -> torch.Tensor:
        B = sensor_embeds.shape[0]
        x = self.latents.expand(B, -1, -1)  # [B, M, H]
        for layer in self.layers:
            x = layer(x, sensor_embeds)
        return self.final_ln(x)


# ============================================================
# 2. Main Model (Route A: VICReg + Resampler)
# ============================================================
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # -------- Config Loading --------
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
        self.stage = int(getattr(args, "stage", 1))

        self.seq_len = int(getattr(args, "seq_len", 96))
        self.patch_len = int(getattr(args, "patch_len", 12))
        self.P = self.seq_len // self.patch_len

        # Augmentation knobs
        self.mask_rate1 = float(getattr(args, "mask_rate1", 0.5))
        self.mask_rate2 = float(getattr(args, "mask_rate2", 0.5))
        self.noise_std1 = float(getattr(args, "noise_std1", 0.01))
        self.noise_std2 = float(getattr(args, "noise_std2", 0.01))

        # -------- LLM Setup --------
        self.max_prompt_len = int(getattr(args, "max_prompt_len", 256))
        self.llama_name = getattr(args, "llama_name", None)
        if self.llama_name is None:
            raise ValueError("args.llama_name is required.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.llama_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_dtype = getattr(args, "llm_dtype", "float16")
        torch_dtype = getattr(torch, llm_dtype, torch.float16)

        print(f"Loading LLM: {self.llama_name}...")
        self.llm = AutoModelForCausalLM.from_pretrained(self.llama_name, torch_dtype=torch_dtype)
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.config.use_cache = False
        self.llm_hidden_size = int(self.llm.config.hidden_size)

        if bool(getattr(args, "freeze_llm", True)):
            self.llm.requires_grad_(False)
            self.llm.eval()

        # -------- Scientific Architecture --------
        # Resampler 维度 (独立于 LLM，建议 768)
        self.resampler_dim = getattr(args, "resampler_dim", 768)
        self.num_latents = int(getattr(args, "num_latents", 16))

        # 1. Patch Embedding (Channel Mixing)
        self.patch_embed = PatchEmbedding(
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            in_channels=self.C,
            embed_dim=self.resampler_dim
        )

        # 2. Deep Resampler
        self.resampler = DeepResampler(
            hidden_size=self.resampler_dim,
            num_latents=self.num_latents,
            depth=getattr(args, "resampler_depth", 4),  # 默认4层
            num_heads=8
        )

        # 3. LLM Projector (Bridge Resampler -> LLM)
        self.llm_proj = nn.Linear(self.resampler_dim, self.llm_hidden_size)

        # 4. Heads
        # Stage 1: VICReg Projection Head (Attached to Resampler)
        proj_dim = int(getattr(args, "proj_dim", 256))
        self.proj_head = nn.Sequential(
            nn.LayerNorm(self.resampler_dim),
            nn.Linear(self.resampler_dim, self.resampler_dim),
            nn.GELU(),
            nn.Linear(self.resampler_dim, proj_dim),
        )

        # Stage 2: Classification Head (Attached to LLM)
        self.cls_head = nn.Linear(self.llm_hidden_size, self.num_class)

        # -------- VICReg Loss Config --------
        self.norm_eps = float(getattr(args, "norm_eps", 1e-5))
        self.vic_inv = float(getattr(args, "vic_inv", 25.0))
        self.vic_var = float(getattr(args, "vic_var", 25.0))
        self.vic_cov = float(getattr(args, "vic_cov", 1.0))
        self.vic_var_eps = float(getattr(args, "vic_var_eps", 1e-4))

        self._prompt_cache = None

    # -------------------------
    # Helper Methods
    # -------------------------
    def _align_seq_len(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        if L != self.seq_len:
            if L > self.seq_len:
                x = x[:, -self.seq_len:, :]
            else:
                pad = self.seq_len - L
                x = torch.cat([torch.zeros(B, pad, C, device=x.device, dtype=x.dtype), x], dim=1)
        return x

    def _make_view(self, x_norm: torch.Tensor, mask_rate: float, noise_std: float) -> torch.Tensor:
        """
        Input: [B, L, C] normalized
        """
        B, L, C = x_norm.shape
        # Patch Masking
        # P = L // patch_len
        patch_mask = (torch.rand(B, self.P, device=x_norm.device) < mask_rate)

        patches = x_norm.view(B, self.P, self.patch_len, C)
        m = patch_mask.view(B, self.P, 1, 1)
        patches = torch.where(m, torch.zeros_like(patches), patches)

        x_view = patches.view(B, L, C)

        # Noise
        if noise_std > 0:
            x_view = x_view + noise_std * torch.randn_like(x_view)

        return x_view

    def _get_text_embeds(self, B: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._prompt_cache is None:
            self._prompt_cache = (
                "You are a sensor assistant. "
                f"Dataset: {self.dataset_key}. "
                "Classify the activity based on the sensor sequence."
            )
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
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        return text_embeds, attn_mask

    def _vicreg_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # invariance (MSE)
        inv = F.mse_loss(z1, z2)

        # variance (std >= 1)
        def var_term(z):
            std = torch.sqrt(z.var(dim=0, unbiased=False) + self.vic_var_eps)
            return torch.mean(F.relu(1.0 - std))

        var = var_term(z1) + var_term(z2)

        # covariance (decorrelate dimensions)
        def cov_term(z):
            z = z - z.mean(dim=0, keepdim=True)
            N, D = z.shape
            cov = (z.T @ z) / max(1, (N - 1))
            off = cov - torch.diag(torch.diag(cov))
            return (off.pow(2).sum() / D)

        cov = cov_term(z1) + cov_term(z2)

        loss = self.vic_inv * inv + self.vic_var * var + self.vic_cov * cov
        meta = {"inv": float(inv.item()), "var": float(var.item()), "cov": float(cov.item())}
        return loss, meta

    # ============================================================
    # Forward Logic (Scientific Two-Stage)
    # ============================================================
    def forward(self, batch_x, padding_mask=None, mode: Optional[str] = None, labels=None):
        if mode is None:
            mode = "pretrain" if self.stage == 1 else "classify"

        if not torch.is_tensor(batch_x):
            batch_x = torch.as_tensor(batch_x)
        x_seq = self._align_seq_len(batch_x)

        # Instance Norm
        mu = x_seq.mean(dim=1, keepdim=True)
        sigma = x_seq.std(dim=1, keepdim=True).clamp_min(self.norm_eps)
        x_norm = (x_seq - mu) / sigma

        # ==========================================
        # Stage 2: Classification (Full Model)
        # ==========================================
        if mode == "classify":
            # 1. Patch Embed & Resample
            # [B, P, D_res]
            z = self.patch_embed(x_norm)
            # [B, M, D_res]
            lat = self.resampler(z)

            # 2. Project to LLM
            # [B, M, D_llm]
            inputs_sensor = self.llm_proj(lat)

            # 3. Text Prompt
            text_embeds, text_mask = self._get_text_embeds(x_seq.shape[0], x_seq.device)
            text_embeds = text_embeds.to(dtype=inputs_sensor.dtype)

            # 4. Concat & LLM
            inputs_embeds = torch.cat([text_embeds, inputs_sensor], dim=1)
            B, M = inputs_sensor.shape[:2]
            sensor_mask = torch.ones(B, M, device=x_seq.device, dtype=text_mask.dtype)
            attention_mask = torch.cat([text_mask, sensor_mask], dim=1)

            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # 5. Pool & Classify
            last_hidden = outputs.hidden_states[-1]
            sensor_out = last_hidden[:, -M:, :]
            pooled = sensor_out.mean(dim=1)
            logits = self.cls_head(pooled)
            return logits

        # ==========================================
        # Stage 1: VICReg Pretraining (Optimization)
        # [Scientific Fix 3]: Skip LLM for efficiency
        # ==========================================

        # 1. Make Views
        v1 = self._make_view(x_norm, self.mask_rate1, self.noise_std1)
        v2 = self._make_view(x_norm, self.mask_rate2, self.noise_std2)

        # 2. Process View 1
        z1 = self.patch_embed(v1)
        lat1 = self.resampler(z1)  # [B, M, D_res]
        y1 = self.proj_head(lat1.mean(dim=1))  # [B, D_proj]

        # 3. Process View 2
        z2 = self.patch_embed(v2)
        lat2 = self.resampler(z2)
        y2 = self.proj_head(lat2.mean(dim=1))

        # 4. VICReg Loss
        loss, meta = self._vicreg_loss(y1.float(), y2.float())

        return loss, None, meta

    def save_wrapper(self, path: str):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_wrapper(self, path: str, map_location="cpu"):
        sd = torch.load(path, map_location=map_location)
        self.load_state_dict(sd, strict=False)



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
    model = Model(configs).to("cuda:0",dtype=torch.float16)
    # model  = model.to()
    x = torch.rand(32,96,15)
    c = model(x.to("cuda:0",dtype=torch.float16),None,None)
    d = 'end'