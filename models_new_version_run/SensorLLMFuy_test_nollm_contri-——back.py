

# models/SensorLLMResampler.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# 禁用 fused attention 以兼容部分旧硬件/环境
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
# 1. Scientific Components (Tokenizer & Resampler)
# ============================================================


class ResamplerLayer(nn.Module):
    """
    标准的 Transformer Decoder Layer 结构用于 Resampler
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
        # 1. Self-Attention (Latents 之间交互)
        q_sa = self.ln_latents(latents)
        latents_out, _ = self.self_attn(q_sa, q_sa, q_sa)
        latents = latents + latents_out

        # 2. Cross-Attention (Latents 查询 Sensor)
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
    [Scientific Improvement]:
    Deep Information Bottleneck.
    Input: Variable length Sensor Embeds [B, P, H]
    Output: Fixed length Latents [B, M, H]
    """

    def __init__(self, hidden_size: int, num_latents: int, depth: int = 4, num_heads: int = 8):
        super().__init__()
        self.num_latents = num_latents
        self.hidden_size = hidden_size

        # Learnable Queries (M个)
        self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_size))
        nn.init.trunc_normal_(self.latents, std=0.02)

        # 堆叠多层
        self.layers = nn.ModuleList([
            ResamplerLayer(hidden_size, num_heads) for _ in range(depth)
        ])

        self.final_ln = nn.LayerNorm(hidden_size)

    def forward(self, sensor_embeds: torch.Tensor) -> torch.Tensor:
        B = sensor_embeds.shape[0]
        # Expand latents for batch
        x = self.latents.expand(B, -1, -1)  # [B, M, H]

        for layer in self.layers:
            x = layer(x, sensor_embeds)

        return self.final_ln(x)

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
class PatchEmbeddingConv(nn.Module):
    """
    Conv stem for HAR inductive bias + patchify.
    Input:  [B, L, C]
    Output: [B, P, D]
    """

    def __init__(self, seq_len: int, patch_len: int, in_channels: int, embed_dim: int):
        super().__init__()
        assert seq_len % patch_len == 0
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Conv stem (acts like TimesNet/TCN low-level feature extractor)
        # Use groups=1 for full channel mixing at low level
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim // 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )

        # Patch projection (tokenize patches in embedding space)
        self.proj = nn.Linear(patch_len * embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # Learnable mask token (for MAE)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B,L,C]
        B, L, C = x.shape
        assert L == self.seq_len, f"expect L={self.seq_len}, got {L}"

        # Conv expects [B,C,L]
        h = self.stem(x.transpose(1, 2))  # [B, D, L]
        h = h.transpose(1, 2)             # [B, L, D]

        # Patchify in embedding space: [B,P,patch_len*D]
        h = torch.reshape(h,(B, self.num_patches, self.patch_len * self.embed_dim))
        # h = h.view()
        h = self.proj(h)                  # [B,P,D]
        h = self.norm(h)
        h = h + self.pos_embed

        if patch_mask is not None:
            mask_tokens = self.mask_token.expand(B, self.num_patches, self.embed_dim)
            w = patch_mask.unsqueeze(-1).type_as(h)
            h = h * (1 - w) + mask_tokens * w

        return h


# ============================================================
# 2. Main Model (InfoNCE pretrain + LLM-free classify)
# ============================================================
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # -------- dataset cfg load --------
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

        # self.C = int(getattr(args, "enc_in", 15))
        # self.num_class = int(getattr(args, "num_class", 12))
        self.stage = int(getattr(args, "stage", 1))

        self.seq_len = int(getattr(args, "seq_len", 200))
        self.patch_len = int(getattr(args, "patch_len", 20))  # make P >= 8
        self.P = self.seq_len // self.patch_len

        self.resampler_dim = int(getattr(args, "resampler_dim", 256))
        self.num_latents = int(getattr(args, "num_latents", min(self.P, 16)))

        # augmentation
        self.mask_rate1 = float(getattr(args, "mask_rate1", 0.3))
        self.mask_rate2 = float(getattr(args, "mask_rate2", 0.5))
        self.noise_std1 = float(getattr(args, "noise_std1", 0.01))
        self.noise_std2 = float(getattr(args, "noise_std2", 0.02))

        # patch embed (conv)
        self.patch_embed = PatchEmbeddingConv(
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            in_channels=self.C,
            embed_dim=self.resampler_dim
        )

        self.resampler = DeepResampler(
            hidden_size=self.resampler_dim,
            num_latents=self.num_latents,
            depth=int(getattr(args, "resampler_depth", 4)),
            num_heads=int(getattr(args, "num_heads", 8))
        )

        proj_dim = int(getattr(args, "proj_dim", 256))
        self.proj_head = nn.Sequential(
            nn.Linear(self.resampler_dim, self.resampler_dim),
            nn.GELU(),
            nn.Linear(self.resampler_dim, proj_dim),
        )

        self.pool = AttnPool(self.resampler_dim)
        self.cls_head = nn.Linear(self.resampler_dim, self.num_class)

        self.norm_eps = float(getattr(args, "norm_eps", 1e-5))
        self.temperature = float(getattr(args, "temperature", 0.1))
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

    def _make_view(self, x_norm: torch.Tensor, mask_rate: float, noise_std: float) -> torch.Tensor:
        B, L, C = x_norm.shape
        P = self.P
        patch_mask = (torch.rand(B, P, device=x_norm.device) < mask_rate)

        patches = x_norm.view(B, P, self.patch_len, C)
        m = patch_mask.view(B, P, 1, 1)
        patches = torch.where(m, torch.zeros_like(patches), patches)

        x_view = patches.view(B, L, C)
        if noise_std > 0:
            x_view = x_view + noise_std * torch.randn_like(x_view)
        return x_view

    def _infonce_loss(self, y1: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.loss_fp32:
            y1 = y1.float()
            y2 = y2.float()

        z1 = F.normalize(y1, dim=1)
        z2 = F.normalize(y2, dim=1)

        B = z1.size(0)
        if B < 2:
            loss = 1.0 - (z1 * z2).sum(dim=1).mean()
            return loss, {"B": float(B), "acc": 0.0}

        logits12 = (z1 @ z2.t()) / max(1e-6, self.temperature)
        logits21 = (z2 @ z1.t()) / max(1e-6, self.temperature)

        targets = torch.arange(B, device=z1.device)
        loss = 0.5 * (F.cross_entropy(logits12, targets) + F.cross_entropy(logits21, targets))

        with torch.no_grad():
            acc = (logits12.argmax(dim=1) == targets).float().mean().item()

        return loss, {"B": float(B), "loss": float(loss.item()), "acc": float(acc)}

    # ============================================================
    # Forward
    # ============================================================
    def forward(self, batch_x, padding_mask=None, mode: Optional[str] = None, labels=None):
        if mode is None:
            mode = "pretrain" if self.stage == 1 else "classify"

        if not torch.is_tensor(batch_x):
            batch_x = torch.as_tensor(batch_x)
        x = self._align_seq_len(batch_x)  # [B, L, C]

        # Instance Norm
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True).clamp_min(self.norm_eps)
        x_norm = (x - mu) / sigma

        # -------------------------
        # Stage 2: LLM-free classify
        # -------------------------
        if mode == "classify":
            sensor_feats = self.patch_embed(x_norm)       # [B, P, D]
            sensor_latents = self.resampler(sensor_feats) # [B, M, D]
            pooled = self.pool(sensor_latents)           # [B, D]
            logits = self.cls_head(pooled)               # [B, num_class]
            return logits

        # -------------------------
        # Stage 1: InfoNCE pretrain (unchanged)
        # -------------------------
        v1 = self._make_view(x_norm, self.mask_rate1, self.noise_std1)
        z1 = self.patch_embed(v1)
        lat1 = self.resampler(z1)
        p1 = self.proj_head(lat1.mean(dim=1))

        v2 = self._make_view(x_norm, self.mask_rate2, self.noise_std2)
        z2 = self.patch_embed(v2)
        lat2 = self.resampler(z2)
        p2 = self.proj_head(lat2.mean(dim=1))

        loss, meta = self._infonce_loss(p1, p2)
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