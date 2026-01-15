# models/SensorLLMFuy.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import yaml
except Exception:
    yaml = None


class Model(nn.Module):
    """
    Scientific HAR-LLM Model (Refined):
    1. Uses Channel-Mixing Patch Embedding (No token explosion).
    2. Uses Soft-Prompting (No vocabulary expansion).
    3. Includes Instance Normalization (RevIN) for stability.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # -----------------------
        # 1. Config Loading
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
        # 2. Dimensions
        # -----------------------
        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 15)))
        self.num_class = int(self.ds_cfg.get("num_labels", getattr(args, "num_class", 12)))
        self.stage = int(getattr(args, "stage", 1))

        self.seq_len = int(getattr(args, "seq_len", 96))
        self.patch_len = int(getattr(args, "patch_len", 12))
        if self.seq_len % self.patch_len != 0:
            raise ValueError(f"seq_len={self.seq_len} must be divisible by patch_len={self.patch_len}")
        self.P = self.seq_len // self.patch_len  # Number of patches

        self.mask_rate = float(getattr(args, "mask_rate", 0.75))

        # -----------------------
        # 3. LLM Setup
        # -----------------------
        self.llama_name = getattr(args, "llama_name", "meta-llama/Llama-2-7b-hf")
        self.max_prompt_len = int(getattr(args, "max_prompt_len", 512))

        self.tokenizer = AutoTokenizer.from_pretrained(self.llama_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load LLM (Frozen by default)
        llm_dtype = getattr(args, "llm_dtype", "float16")
        torch_dtype = getattr(torch, llm_dtype, torch.float16)

        self.llm = AutoModelForCausalLM.from_pretrained(self.llama_name, torch_dtype=torch_dtype)
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.config.use_cache = False

        # Freeze LLM weights to preserve pretrained knowledge
        self.llm.requires_grad_(False)
        self.llm.eval()

        self.H = int(self.llm.config.hidden_size)

        # -----------------------
        # 4. Scientific Projector (The "Adapter")
        # -----------------------
        # Channel Mixing: Input is (Patch_Len * C) -> Project to H
        # This dramatically reduces sequence length (from P*C to P)
        self.patch_dim = self.patch_len * self.C

        self.enc_proj = nn.Linear(self.patch_dim, self.H)
        self.enc_norm = nn.LayerNorm(self.H)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.H))
        nn.init.normal_(self.mask_token, std=0.02)

        # -----------------------
        # 5. Heads
        # -----------------------
        # Pretrain Head: Project hidden state back to patch size
        self.recon_head = nn.Linear(self.H, self.patch_dim)

        # Classification Head
        self.cls_head = nn.Linear(self.H, self.num_class)

        # Prompt cache
        self._prompt_text_cache: Optional[str] = None

        # Stability
        self.norm_eps = 1e-5

    # ============================================================
    # Instance Normalization (RevIN style)
    # ============================================================
    def _instance_norm(self, x):
        # x: [B, L, C]
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(self.norm_eps)
        return (x - mean) / std, mean, std

    # ============================================================
    # Patchify (Channel Mixing)
    # ============================================================
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        returns: [B, P, patch_dim] where patch_dim = patch_len * C
        """
        B, L, C = x.shape
        # 1. Reshape to patches
        x = x.view(B, self.P, self.patch_len, C)
        # 2. Flatten patch_len and C
        x = x.view(B, self.P, self.patch_len * C)
        return x

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, P, patch_dim]
        returns: [B, L, C]
        """
        B, P, _ = x.shape
        x = x.view(B, P, self.patch_len, self.C)
        x = x.view(B, self.seq_len, self.C)
        return x

    # ============================================================
    # Prompt Building
    # ============================================================
    def _build_prompt(self, B, device):
        if self._prompt_text_cache is None:
            # We use a placeholder <SENSOR> that will be replaced by embeddings
            self._prompt_text_cache = (
                "You are a sensor assistant. "
                f"Dataset: {self.dataset_key}. "
                "Analyze the following sensor data sequence: "
            )

        prompts = [self._prompt_text_cache] * B
        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_len,
            return_tensors="pt"
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    # ============================================================
    # Forward
    # ============================================================
    def forward(self, batch_x, padding_mask=None, mode: Optional[str] = None, labels=None):
        if mode is None:
            mode = "pretrain" if self.stage == 1 else "classify"

        # 1. Input Validation
        if not torch.is_tensor(batch_x):
            batch_x = torch.as_tensor(batch_x)

        # 2. Align & Normalize (Critical for MSE)
        B, L, C = batch_x.shape
        if L != self.seq_len:
            # simple truncation/pad logic
            if L > self.seq_len:
                batch_x = batch_x[:, :self.seq_len, :]
            else:
                batch_x = F.pad(batch_x, (0, 0, 0, self.seq_len - L))

        x_norm, mu, std = self._instance_norm(batch_x)

        # 3. Patchify
        # [B, P, patch_dim]
        patches = self._patchify(x_norm)

        # 4. Masking (Only for Pretrain)
        if mode == "pretrain":
            mask = torch.rand(B, self.P, device=batch_x.device) < self.mask_rate
            # Ensure at least one patch is visible/masked to prevent NaNs
            mask[:, 0] = False
        else:
            mask = torch.zeros(B, self.P, device=batch_x.device, dtype=torch.bool)

        # 5. Project to LLM Dimension
        # [B, P, H]
        sensor_embeds = self.enc_proj(patches)
        sensor_embeds = self.enc_norm(sensor_embeds)

        # Apply mask tokens
        if mode == "pretrain":
            mask_token = self.mask_token.expand(B, self.P, self.H)
            # Replace masked positions with learnable mask token
            sensor_embeds = torch.where(mask.unsqueeze(-1), mask_token, sensor_embeds)

        # 6. Combine with Text Prompt (Soft Prompting)
        # Text: [B, T_text] -> [B, T_text, H]
        input_ids, attn_mask = self._build_prompt(B, batch_x.device)
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # Concat: [Text, Sensor]
        # [B, T_text + P, H]
        inputs_embeds = torch.cat([text_embeds, sensor_embeds], dim=1)

        # Extend attention mask
        sensor_attn_mask = torch.ones(B, self.P, device=batch_x.device, dtype=attn_mask.dtype)
        full_attn_mask = torch.cat([attn_mask, sensor_attn_mask], dim=1)

        # 7. LLM Forward
        # We only need the output, no gradients for LLM backbone
        with torch.no_grad():
            outputs = self.llm(
                inputs_embeds=inputs_embeds.to(dtype=self.llm.dtype),
                attention_mask=full_attn_mask,
                output_hidden_states=True,
                use_cache=False
            )

        hidden_states = outputs.hidden_states[-1]  # [B, Total_Len, H]

        # Extract only the sensor part (last P tokens)
        sensor_output = hidden_states[:, -self.P:, :]  # [B, P, H]

        # ----------------------------------------
        # Branch 1: Pretrain (MAE)
        # ----------------------------------------
        if mode == "pretrain":
            # Reconstruct patches
            pred_patches = self.recon_head(sensor_output)  # [B, P, patch_dim]

            # Loss is calculated ONLY on masked patches
            loss_mse = (pred_patches - patches) ** 2
            loss_mse = loss_mse.mean(dim=-1)  # [B, P]

            # Apply mask (0 loss for visible patches)
            loss_mse = (loss_mse * mask.float()).sum() / (mask.float().sum() + 1e-6)

            meta = {
                "mask_rate_real": float(mask.float().mean().item()),
                "loss": loss_mse.item()
            }
            return loss_mse, mask, meta

        # ----------------------------------------
        # Branch 2: Classify
        # ----------------------------------------
        else:
            # Simple Pooling (Mean over all sensor patches)
            # You can also use the last token, but mean is generally more stable for time series
            pooled = sensor_output.mean(dim=1)  # [B, H]
            logits = self.cls_head(pooled)  # [B, Num_Class]
            return logits

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
    x = torch.rand(32,96,15)
    c = model(x.to("cuda:0",dtype=torch.float16),None,None)
    d = 'end'