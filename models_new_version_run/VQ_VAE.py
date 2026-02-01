import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch import Tensor
from typing import Any, Dict, Optional, Tuple, List


try:
    import yaml
except Exception:
    yaml = None

# optional: online LLM teacher
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None

import torch
import torch.nn.functional as F

def _pad_to_multiple(x: torch.Tensor, multiple: int, pad_value: float = 0.0):
    """
    x: [B, L, C]
    return: x_pad [B, L_pad, C], L_orig
    """
    B, L, C = x.shape
    L_pad = ((L + multiple - 1) // multiple) * multiple
    if L_pad == L:
        return x, L
    pad_len = L_pad - L
    pad = x.new_full((B, pad_len, C), pad_value)
    return torch.cat([x, pad], dim=1), L

def _pad_mask_to_len(mask: torch.Tensor, L_pad: int):
    """
    mask: [B, L] (bool or 0/1), True/1 means valid
    return: mask_pad [B, L_pad] bool
    """
    B, L = mask.shape
    mask_bool = mask.bool()
    if L == L_pad:
        return mask_bool
    pad_len = L_pad - L
    pad = torch.zeros((B, pad_len), device=mask.device, dtype=torch.bool)
    return torch.cat([mask_bool, pad], dim=1)

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """
    pred/target: [B, L, C]
    mask: [B, L] bool (True = valid)
    """
    mask = mask.unsqueeze(-1)  # [B,L,1]
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask
    denom = mask.sum() * pred.shape[-1]
    return diff2.sum() / denom.clamp_min(1.0)

def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    mask = mask.unsqueeze(-1)
    diff = (pred - target).abs() * mask
    denom = mask.sum() * pred.shape[-1]
    return diff.sum() / denom.clamp_min(1.0)

def masked_frequency_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """
    频域 loss 在严格 mask 下不太“完美”（因为 rfft 会把全序列混在一起），
    但最简单可控的做法是：先把无效位置置 0，再做 rfft。
    pred/target: [B,L,C], mask: [B,L] bool
    """
    m = mask.unsqueeze(-1)  # [B,L,1]
    pred0 = pred * m
    tgt0  = target * m
    pred_fft = torch.fft.rfft(pred0, dim=1)   # dim=1 是时间维
    tgt_fft  = torch.fft.rfft(tgt0, dim=1)
    return F.mse_loss(pred_fft.abs(), tgt_fft.abs())


class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).to(device))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):

        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                                  keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()  # (N, DIM, T)

        return x_d, commit_loss, perplexity


class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(Quantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        N, width, T = z.shape
        z = self.preprocess(z)
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q - z.detach()) ** 2) + self.beta * \
               torch.mean((z_q.detach() - z) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        z_q = z_q.view(N, T, -1).permute(0, 2, 1).contiguous()  # (N, DIM, T)

        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        return z_q, loss, perplexity

    def quantize(self, z):
        assert z.shape[-1] == self.e_dim

        # B x V
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def dequantize(self, indices):
        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.e_dim,)).contiguous()
        return z_q

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x


class QuantizeReset(nn.Module):
    def __init__(self, nb_code, code_dim):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.reset_codebook()
        self.codebook = nn.Parameter(torch.randn(nb_code, code_dim))

    def reset_codebook(self):
        self.init = False
        self.code_count = None

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = nn.Parameter(out[:self.nb_code])
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    def update_codebook(self, x, code_idx):

        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_count = code_count  # nb_code
        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()

        self.codebook.data = usage * self.codebook.data + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                                  keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x):
        N, width, T = x.shape
        # Preprocess
        x = self.preprocess(x)
        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)
        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)
        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()  # (N, DIM, T)

        return x_d, commit_loss, perplexity


class QuantizeEMA(nn.Module):
    def __init__(self, nb_code, code_dim, mu):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):

        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = code_update
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                                  keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()  # (N, DIM, T)

        return x_d, commit_loss, perplexity


class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None):
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)

        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()

        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0, )

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()

        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm)
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

# Encoder 和 Decoder 类保持原样即可，它们的逻辑是通用的
class Encoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate,
                 activation='relu', norm=None):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate,
                 activation='relu', norm=None):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


def frequency_loss(pred, target):
    # 将时间序列转到频域 (FFT)
    # dim=-2 是时间维度
    pred_fft = torch.fft.rfft(pred, dim=-2)
    target_fft = torch.fft.rfft(target, dim=-2)

    # 计算频域幅度的差异
    loss = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))
    return loss
class IMU_VQ_Model(nn.Module):
    def __init__(self,
                 args) -> None:

        super().__init__()
        self.args = args

        self.device = args.device
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
        self.d_model = self.args.d_model
        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 15)))
        self.num_class = int(self.ds_cfg.get("num_labels", getattr(args, "num_class", 12)))
        self.down_sampling_layers = self.args.down_sampling_layers

        self.input_dim=self.C #  这里填你的 nvar (例如 6)
        self.code_num=  512  # 词典大小 (有多少个不同的 token)
        self.output_emb_width=  512  # Encoder输出的通道数
        self.code_dim=  512  # 编码后的向量维度
        self.down_t=  3  # 下采样次数 (时间轴压缩倍率 = 2^down_t)
        self.stride_t=  2  # 步长
        self.width=  512  # 中间隐藏层通道数
        self.depth=  3  # ResNet块的深度
        self.dilation_growth_rate=  3
        self.norm=None
        self.activation=  "relu"
        self.quantizer=  "ema_reset"  # 推荐用 ema_reset 防止码本坍塌





        # 1. 编码器：把 IMU 数据压缩成特征
        self.encoder = Encoder(input_emb_width=self.input_dim,  # 输入维度 (nvar)
                               output_emb_width=self.output_emb_width,
                               down_t=self.down_t,
                               stride_t=self.stride_t,
                               width=self.width,
                               depth=self.depth,
                               dilation_growth_rate=self.dilation_growth_rate,
                               activation=self.activation,
                               norm=self.norm)

        # 2. 解码器：把特征还原成 IMU 数据
        self.decoder = Decoder(input_emb_width=self.input_dim,  # 输出维度 (nvar) - 注意这里要还原回原始维度
                               output_emb_width=self.output_emb_width,
                               down_t=self.down_t,
                               stride_t=self.stride_t,
                               width=self.width,
                               depth=self.depth,
                               dilation_growth_rate=self.dilation_growth_rate,
                               activation=self.activation,
                               norm=self.norm)

        # 3. 量化器：把连续特征变成离散 Code ID
        if self.quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(self.code_num, self.code_dim, mu=0.99)
        elif self.quantizer == "orig":
            self.quantizer = Quantizer(self.code_num, self.code_dim, beta=1.0)
        elif self.quantizer == "ema":
            self.quantizer = QuantizeEMA(self.code_num, self.code_dim, mu=0.99)
        elif self.quantizer == "reset":
            self.quantizer = QuantizeReset(self.code_num, self.code_dim)

    def preprocess(self, x):
        # 你的输入: (batch, seq_len, nvar)
        # Conv1d 需要: (batch, nvar, seq_len)
        # 所以我们需要 permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # 还原回: (batch, seq_len, nvar)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features, padding_mask=None, mode="pretrain"):
        """
        features: [B, L, C]
        padding_mask: [B, L] (可选) True/1 表示有效位置
        """
        B, L, C = features.shape

        # 1) pad 到 2^down_t 的倍数，避免 100->96 的结构性截断
        multiple = 2 ** int(self.down_t)  # down_t=3 => 8
        x_in, L_orig = _pad_to_multiple(features, multiple=multiple, pad_value=0.0)  # [B, L_pad, C]
        L_pad = x_in.shape[1]

        # 2) mask 对齐到 pad 后长度：pad 部分强制无效
        if padding_mask is None:
            mask = torch.ones((B, L_orig), device=features.device, dtype=torch.bool)
        else:
            mask = padding_mask.bool()
        mask = _pad_mask_to_len(mask, L_pad)  # [B, L_pad]，pad 部分是 False

        # 3) 走你的编码-量化-解码
        x_conv_in = self.preprocess(x_in)  # [B, C, L_pad]
        x_encoder = self.encoder(x_conv_in)  # [B, D, L_pad/8]
        x_quantized, qua_loss, perplexity = self.quantizer(x_encoder)
        x_decoder = self.decoder(x_quantized)  # [B, C, L_pad] （理想情况下回到 L_pad）
        x_out_pad = self.postprocess(x_decoder)  # [B, L_pad, C]

        # 4) crop 回原长度（对外输出严格等于输入长度）
        x_out = x_out_pad[:, :L_orig, :]  # [B, L_orig, C]
        mask_crop = mask[:, :L_orig]  # [B, L_orig]

        # 5) 只在有效位置算 loss（pad 不参与）
        recon_loss = masked_mse(x_out, features, mask_crop)
        recon_loss_l1 = masked_l1(x_out, features, mask_crop)
        freq_loss = masked_frequency_loss(x_out, features, mask_crop)

        # finel_loss = recon_loss + recon_loss_l1 + freq_loss + qua_loss
        if self.args.loss_style == "qua":
            finel_loss = qua_loss
        elif self.args.loss_style == "freq":
            finel_loss = freq_loss
        elif self.args.loss_style == "recon":
            finel_loss = recon_loss + recon_loss_l1
        elif self.args.loss_style == "qua+freq":
            finel_loss =qua_loss+freq_loss
        elif self.args.loss_style == "qua+recon":
            finel_loss = qua_loss + recon_loss + recon_loss_l1
        elif self.args.loss_style == "freq+recon":
            finel_loss = freq_loss + recon_loss + recon_loss_l1
        else:
            finel_loss = recon_loss + recon_loss_l1 + freq_loss + qua_loss

        # finel_loss =  qua_loss

        return finel_loss, x_out, {
            "qua_loss": qua_loss,
            "recon_loss": recon_loss,
            "recon_loss_l1": recon_loss_l1,
            "freq_loss": freq_loss,
            "perplexity": perplexity,
            "finel_loss": finel_loss,
            "L_orig": L_orig,
            "L_pad": L_pad,
        }

    def get_token_ids(self, features: Tensor):
        """
        【重要】做 LLM 训练时用这个方法！
        输入: IMU 数据 (Batch, Seq_Len, N_Var)
        输出: Token ID 序列 (Batch, Seq_Len_Compressed)
        """
        N, T, _ = features.shape
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)  # (B, C, T_compressed)

        # 调整形状以适应量化器
        x_encoder = x_encoder.permute(0, 2, 1)  # (B, T_compressed, C)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (B*T, C)

        # 获取 ID
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)  # (B, T_compressed)

        return code_idx

    def decode_from_ids(self, ids: Tensor):
        """
        验证时用，看 Token 对应的动作长啥样
        输入: Token ID (Batch, T_compressed)
        输出: IMU 数据
        """
        x_d = self.quantizer.dequantize(ids)
        x_d = x_d.view(ids.shape[0], -1, self.code_dim).permute(0, 2, 1).contiguous()
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


    def save_wrapper(self, path: str):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_wrapper(self, path: str, map_location="cpu"):
        sd = torch.load(path, map_location=map_location)
        self.load_state_dict(sd, strict=False)




# --task_name=lora
# --task_name=classification
# --task_name=vqvae
# --is_training=1
# --root_path="D:/fuy/MyCode/SensorLLMLib/datasets/MHEALTHDATASET"
# --root_path="D:/fuy/MyCode/SensorLLMLib/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
# --model_id=UCIHAR
# --run_id="alignment_weight"
# --datasets=UCIHAR
# --model="SensorLoRA"
# --model="SensorLLMFuy_test_withllm_mae_vqvae"
# --model="VQVAE"
# --data=UCIHAR
# --dataset_key=ucihar
# --seq_len=200
# --patch_len=64
# --stride=64
# --stage=1
# --batch_size=32
# --llama_name="D:\fuy\MyCode\Llama-3.2-1B"
# --learning_rate=0.001
# --train_epochs=10
# --num_workers=0
# --vqvae_path=qua_recon_path
# --test_subjects="subject1,subject3,subject6"
# --mask_rate=0.75
# --itr=1


import matplotlib.pyplot as plt
import numpy as np


def validate_vqvae(model, test_loader, device="cuda"):
    """
    验证 VQ-VAE 效果的核心函数
    """
    model.eval()
    model.to(device)

    total_mse = 0
    total_samples = 0
    all_tokens = []

    # 1. 只需要跑一个 Batch 来看效果即可，或者跑整个测试集
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):  # 假设 dataloader 返回 (data, label)
            data = data.to(device).float()

            # 前向传播
            # 注意：你的 forward 返回值是: finel_loss, x_out, metrics
            loss, x_recon, metrics = model(data)

            # 收集重建误差
            # x_recon: [B, L, C], data: [B, L, C]
            mse = F.mse_loss(x_recon, data, reduction='sum')
            total_mse += mse.item()
            total_samples += data.numel()

            # 收集 Token 使用情况 (检查是否坍塌)
            # 使用你写的 get_token_ids 方法
            token_ids = model.get_token_ids(data)  # [B, T_compressed]
            all_tokens.append(token_ids.cpu().numpy().flatten())

            # --- 可视化第一个 Batch 的第一个样本 ---
            if batch_idx == 0:
                visualize_reconstruction(data, x_recon, sample_idx=0, channel_idx=0)

            # 为了演示，只跑一个 batch 就 break，实际验证可以跑完
            break

            # 2. 统计 Token 利用率
    all_tokens = np.concatenate(all_tokens)
    unique_tokens = len(np.unique(all_tokens))
    avg_mse = total_mse / total_samples

    print(f"\n====== 验证报告 ======")
    print(f"📉 平均 MSE Loss: {avg_mse:.6f}")
    print(f"🧩 码本总大小: {model.code_num}")
    print(f"✅ 实际激活 Token 数: {unique_tokens} (如果这个数很小，比如 <10，说明发生 Codebook Collapse)")
    print(f"📊 当前 Perplexity: {metrics['perplexity'].item():.4f}")


def visualize_reconstruction(real_data, recon_data, sample_idx=0, channel_idx=0):
    """
    修改版：保存图片而不是显示，避免 PyCharm Backend 报错
    """
    real = real_data[sample_idx, :, channel_idx].cpu().numpy()
    recon = recon_data[sample_idx, :, channel_idx].cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(real, label='Original (GT)', color='black', alpha=0.7)
    plt.plot(recon, label='Reconstruction', color='red', linestyle='--', alpha=0.8)
    plt.title(f"Reconstruction Check (Sample {sample_idx}, Channel {channel_idx})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- 修改这里 ---
    # 原代码: plt.show()
    # 新代码: 保存到当前目录下的 vq_check.png
    save_path = "vq_check.png"
    plt.savefig(save_path)
    plt.close()  # 关闭图表释放内存
    print(f"🖼️ 验证图片已保存至: {os.path.abspath(save_path)}")

# ==========================================
# 如何调用 (示例)
# ==========================================
def vos():
    # 1. 模拟参数 (因为你的模型需要 args)
    class Args:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dataset_key = "mhealth"  # 或者 ucihar
        d_model = 512
        down_sampling_layers = 3
        loss_style = "recon"  # 确保能跑通
        # 添加其他必要的参数...


    args = Args()

    # 2. 初始化模型
    # 注意：确保 input_dim 和你的数据一致，例如 6 或 9
    model = IMU_VQ_Model(args)

    # 3. 加载权重 (这一步很重要！)
    # model.load_wrapper("你的权重路径.pth")

    # 4. 制造假数据或使用你的 DataLoader
    # 假设数据格式是 [Batch=32, Len=96, Channel=6]
    dummy_data = torch.randn(32, 200, model.input_dim)
    dummy_loader = [(dummy_data, None)]  # 模拟 DataLoader

    print("开始验证 VQ-VAE...")
    validate_vqvae(model, dummy_loader, device=args.device)
if __name__ == '__main__':
    vos()



    batch_imu = torch.randn(32,96,7)
    model = IMU_VQ_Model(input_dim=7)
    criterion_mse = nn.MSELoss()
    # criterion_freq = FrequencyAwareLoss()  # 你上一篇论文的精华

    # 训练循环
    x_in = batch_imu  # (B, T, 6)
    x_recon, loss_commit, perplexity = model(x_in)

    # 1. 时域重建 Loss
    loss_mse = criterion_mse(x_recon, x_in)

    # 2. 【关键】频域重建 Loss (你的创新点)
    # loss_freq = criterion_freq(x_recon, x_in)
    res = model.get_token_ids(x_in)
    aaa = model.decode_from_ids(res)
    loss_freq = 0

    # 总 Loss
    loss_total = loss_mse + 0.1 * loss_freq + loss_commit
    loss_total.backward()