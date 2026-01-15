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

# models/SensorLLMResampler.py (Route A - Scientific MAE)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# 禁用 fused attention 以兼容性优先
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

try:
    import yaml
except Exception:
    yaml = None


# ============================================================
# 1. Scientific Components (Tokenizer, Resampler, Decoder)
# ============================================================
PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"


class PatchEmbedding(nn.Module):
    """
    [Scientific Fix]:
    1. Channel Independent (通道独立): 更好地捕捉时序特征。
    2. Learnable Mask Token: 明确告诉模型“这里缺失了”，而不是“这里是0”。
    """

    def __init__(self, seq_len: int, patch_len: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # proj: 此时只处理单通道的时间片段 (patch_len * 1) -> embed_dim
        # 我们最后把 embed_dim 拆回 (embed_dim // in_channels) 或者直接映射
        # 为了兼容你的 Resampler，我们还是保持输出 [B, P, D]

        # 简单起见，我们还是用 Channel Mixing 但加上 Mask Token
        self.proj = nn.Linear(patch_len * in_channels, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # === [关键修改] 可学习的 Mask Token ===
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, L, C]
        B, L, C = x.shape

        # 1. Patchify
        x = x.view(B, self.num_patches, self.patch_len * C)
        x = self.proj(x)  # [B, P, D]
        x = self.norm(x)
        x = x + self.pos_embed  # Add pos first

        # 2. Apply Mask Token (If training MAE)
        if patch_mask is not None:
            # patch_mask: [B, P], True means Masked (Hidden)
            B, P, D = x.shape
            # 扩展 mask token 到 [B, P, D]
            mask_tokens = self.mask_token.expand(B, P, D)
            # 制作 mask 索引: [B, P, 1]
            w = patch_mask.unsqueeze(-1).type_as(x)

            # 如果 masked(True=1)，使用 mask_token；否则保持原样 x
            x = x * (1 - w) + mask_tokens * w

        return x


class ResamplerLayer(nn.Module):
    """
    Standard Resampler Block: Self-Attn -> Cross-Attn -> FFN
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
    [Scientific Fix 2]: Deep Bottleneck
    """

    def __init__(self, hidden_size: int, num_latents: int, depth: int = 4, num_heads: int = 8):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_size))
        nn.init.trunc_normal_(self.latents, std=0.02)

        self.layers = nn.ModuleList([
            ResamplerLayer(hidden_size, num_heads) for _ in range(depth)
        ])
        self.final_ln = nn.LayerNorm(hidden_size)

    def forward(self, sensor_embeds: torch.Tensor) -> torch.Tensor:
        B = sensor_embeds.shape[0]
        x = self.latents.expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, sensor_embeds)
        return self.final_ln(x)


class MAEDecoder(nn.Module):
    """
    [Scientific Fix 3]: Lightweight Decoder for Pretraining.
    它从 Latents 重建原始 Patch，不经过 LLM。
    """

    def __init__(self, hidden_size: int, num_patches: int, patch_dim: int, num_heads: int = 4, depth: int = 2):
        super().__init__()
        self.num_patches = num_patches

        # Positional Queries: 告诉 Decoder 我们要重建哪个位置的 Patch
        self.pos_queries = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_queries, std=0.02)

        # Decoder Layers (Simple Transformer Decoder)
        # 这里用 Cross-Attn: Query=Pos, Key/Value=Latents
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True, norm_first=True)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        self.pred_head = nn.Linear(hidden_size, patch_dim)  # Output: Patch_Len * C

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: [B, M, H]
        B = latents.shape[0]

        # Construct Queries: [B, P, H]
        queries = self.pos_queries.expand(B, -1, -1)

        x = queries
        for layer in self.layers:
            # tgt=x, memory=latents
            x = layer(tgt=x, memory=latents)

        x = self.norm(x)
        pred = self.pred_head(x)  # [B, P, Patch_Len * C]
        return pred

class AttnPool(nn.Module):
    """Lightweight attention pooling: [B,M,D] -> [B,D]"""
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, max(1, dim // 2)),
            nn.GELU(),
            nn.Linear(max(1, dim // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.score(x)                 # [B, M, 1]
        w = torch.softmax(w, dim=1)       # [B, M, 1]
        return (x * w).sum(dim=1)         # [B, D]


# ============================================================
# 2. Main Model (MAE pretrain + LLM-free classify)
# ============================================================
class Model(nn.Module):
    """
    Stage 1: MAE Pretraining. Sensor -> Resampler -> Decoder -> Recon.
    Stage 2: LLM-free Classification. Sensor -> Resampler -> Pool -> Classify.
    """

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
        self.mask_rate = float(getattr(args, "mask_rate", 0.75))

        # -------- Scientific Architecture --------
        self.resampler_dim = int(getattr(args, "resampler_dim", 768))
        self.num_latents = int(getattr(args, "num_latents", 64))

        # 1) Patch Embedding
        self.patch_embed = PatchEmbedding(
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            in_channels=self.C,
            embed_dim=self.resampler_dim
        )

        # 2) Deep Resampler
        self.resampler = DeepResampler(
            hidden_size=self.resampler_dim,
            num_latents=self.num_latents,
            depth=getattr(args, "resampler_depth", 4),
            num_heads=8
        )

        # 3) MAE Decoder (Stage 1)
        self.mae_decoder = MAEDecoder(
            hidden_size=self.resampler_dim,
            num_patches=self.P,
            patch_dim=self.patch_len * self.C,
            depth=2
        )

        # 4) LLM-free classifier head (Stage 2)
        self.pool = AttnPool(self.resampler_dim)
        self.cls_head = nn.Linear(self.resampler_dim, self.num_class)

        # Misc
        self.norm_eps = float(getattr(args, "norm_eps", 1e-5))
        self.loss_fp32 = bool(getattr(args, "loss_fp32", True))

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

    # ============================================================
    # Forward
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

        # ---------------------------------------------
        # Stage 2: LLM-free Classification
        # ---------------------------------------------
        if mode == "classify":
            z = self.patch_embed(x_norm)      # [B, P, D]
            lat = self.resampler(z)           # [B, M, D]
            pooled = self.pool(lat)           # [B, D]
            logits = self.cls_head(pooled)    # [B, num_class]
            return logits

        # ---------------------------------------------
        # Stage 1: MAE Pretraining (keep your original logic)
        # ---------------------------------------------
        B, L, C = x_norm.shape
        patch_mask = self._random_patch_mask(B, self.P, x_norm.device)  # [B, P] bool

        # IMPORTANT: keep your MAE mask-token path
        z = self.patch_embed(x_norm, patch_mask=patch_mask)
        latents = self.resampler(z)  # [B, M, D]

        pred_patches = self.mae_decoder(latents)  # [B, P, patch_len*C]
        target_patches = x_norm.view(B, self.P, self.patch_len * C)

        if self.loss_fp32:
            pred_patches = pred_patches.float()
            target_patches = target_patches.float()

        loss = (pred_patches - target_patches) ** 2
        loss = loss.mean(dim=-1)  # [B, P]
        loss = (loss * patch_mask.float()).sum() / (patch_mask.float().sum() + 1e-6)

        meta = {
            "mask_rate": float(patch_mask.float().mean().item()),
            "loss": float(loss.item())
        }
        return loss, patch_mask, meta

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
    model = Model(configs).to("cuda:0")
    x = torch.rand(32,96,15)
    c = model(x.to("cuda:0"),None,None)
    d = 'end'