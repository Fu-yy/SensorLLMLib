# models/SensorLLMResampler.py  (Route A)
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from models_new_version_run.VQ_VAE import IMU_VQ_Model

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

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml


# ==========================================
# 1. 辅助类与函数
# ==========================================

@dataclass
class TeacherOut:
    probs: torch.Tensor  # [B, K] Teacher对mask位置的预测分布
    valid: torch.Tensor  # [B] 标记该样本是否成功生成了guidance


def ids_to_special_tokens(token_ids: torch.Tensor, K: int) -> List[List[str]]:
    """
    将 Token ID 矩阵转换为字符串列表
    Example: [[1, 512, 2], ...] -> [["<P001>", "<PMASK>", "<P002>"], ...]
    假设 512 (或 K) 是 mask token 的 id
    """
    B, T = token_ids.shape
    batch_tokens = []
    mask_token_id = K  # 假设 Mask Token ID 是 K

    for b in range(B):
        seq = []
        for t in range(T):
            tid = token_ids[b, t].item()
            if tid == mask_token_id:
                seq.append("<PMASK>")
            else:
                seq.append(f"<P{tid:03d}>")
        batch_tokens.append(seq)
    return batch_tokens


# ==========================================
# 2. Primitive LLM Teacher (冻结的大模型)
# ==========================================

class PrimitiveLLMTeacher(nn.Module):
    """
    Frozen causal LLM teacher.
    It takes a sequence with <PMASK> and predicts the probability distribution
    of the primitive tokens {<P000>..<P{K-1}>} at the mask position.
    """

    def __init__(self, llm_name_or_path: str, K: int, device: torch.device, max_len: int = 512):
        super().__init__()
        self.K = int(K)
        self.device = device
        self.max_len = int(max_len)

        print(f"[Teacher] Loading LLM from {llm_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, use_fast=False)

        # 确保有 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 1. 添加 Special Tokens: <P000>...<P511> 和 <PMASK>
        # 格式化为 3位数字，例如 <P005>
        prim_tokens = [f"<P{i:03d}>" for i in range(self.K)] + ["<PMASK>"]
        num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": prim_tokens})

        # 2. 加载模型
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name_or_path)
        if num_added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))

        # 3. 冻结并移至设备
        self.llm.to(device).eval()
        for p in self.llm.parameters():
            p.requires_grad = False

        # 4. 预缓存 primitive token ids，用于从 logits 中提取
        # 注意：这里我们只关心 <P000> 到 <PK-1> 的 ID
        prim_ids_list = [self.tokenizer.convert_tokens_to_ids(f"<P{i:03d}>") for i in range(self.K)]
        self.register_buffer("prim_token_ids", torch.tensor(prim_ids_list, dtype=torch.long))

    def build_prompts(self, masked_ids: torch.Tensor) -> List[str]:
        """
        根据 masked_ids 构建 LLM 的 Prompt。
        masked_ids: [B, T], 其中 mask 的位置值为 self.K
        """
        prompts = []
        token_strs = ids_to_special_tokens(masked_ids, self.K)  # [B, List[str]]

        for b_idx, seq_tokens in enumerate(token_strs):
            seq_str = " ".join(seq_tokens)
            # 构建 Prompt。这里你可以根据论文需求调整 Prompt Engineering
            prompt = (
                "You are an expert in sensor activity analysis.\n"
                f"Sequence: {seq_str}\n"
                "Task: Predict the sensor token that should replace <PMASK>.\n"
                "Answer:"
            )
            prompts.append(prompt)
        return prompts

    def forward(self, masked_ids: torch.Tensor) -> TeacherOut:
        """
        Args:
            masked_ids: [B, T] 包含 mask 的 token 序列
        Returns:
            TeacherOut: 包含每个样本针对 <PMASK> 的预测分布
        """
        prompts = self.build_prompts(masked_ids)

        # Tokenize
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len
        ).to(self.device)

        # Inference
        with torch.no_grad():
            out = self.llm(**enc)
            # 取最后一个 token 的 logits (假设 LLM 紧接着 prompt 预测 mask 的内容)
            # 注意：对于 Causal LM，通常是取 prompt 最后一个 token 对应的输出作为 next token prediction
            logits = out.logits[:, -1, :]  # [B, VocabSize]

        # 提取 Primitive Tokens 的 logits
        # self.prim_token_ids 是 [K]
        # prim_logits: [B, K]
        prim_logits = logits.index_select(dim=-1, index=self.prim_token_ids)

        # 转换为概率分布 (Temperature 可以控制分布平滑度)
        probs = torch.softmax(prim_logits, dim=-1)

        # 标记 Valid (这里简单全部视为有效，实际可根据逻辑过滤)
        valid = torch.ones(masked_ids.size(0), dtype=torch.bool, device=self.device)

        return TeacherOut(probs=probs, valid=valid)


# ==============================================================================
# Part 4: SoftLlamaTeacher (Simplified for integration)
# ==============================================================================

class StudentTransformer(nn.Module):
    """
    轻量级的 Transformer，用于学习 mask 补全任务。
    最后部署时只保留这个。
    """

    def __init__(self,codebook_weights , num_tokens, dim_model=256, nhead=4, num_layers=4, max_len=512):
        super().__init__()
        # codebook_weight = vq_net.quantizer.embedding.weight.data.clone()

        # 假设 saved_codebook 是 [K, vq_dim]
        # codebook_weights = torch.load(code_path, map_location="cpu")
        # 根据你的保存格式，可能是 ckpt['embedding'] 或直接是 ckpt
        # codebook_weights = ckpt['model']['quantizer.embedding.weight']  # 举例，需根据实际情况调整
        # num_codes, code_dim = codebook_weights.shape
        # +1 用于 Mask Token
        self.embedding = nn.Embedding(num_tokens+1, dim_model)

        # 1. 创建一个新的权重矩阵，包含 mask token
        # 我们用xavier_normal或者zeros初始化它，作为起步
        new_weight = torch.empty(num_tokens + 1, dim_model)
        nn.init.xavier_normal_(new_weight)

        # 2. 把前 512 个位置替换成 codebook 的权重
        # 注意：前提是 dim_model 必须等于 codebook 的 dim
        new_weight[:num_tokens] = codebook_weights

        # 3. 第 513 个位置 (index 512) 是 Mask Token，保留随机初始化让它去学
        # self.mask_token_id = num_tokens

        # 4. 赋值给 Embedding 层
        self.embedding.weight.data.copy_(new_weight)
        #
        # # 选项：你可以选择是否冻结它 (False = 允许微调，True = 锁死)
        # self.embedding.weight.requires_grad = True

        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, dim_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=nhead, dim_feedforward=dim_model * 4,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Linear(dim_model, num_tokens)  # 输出 K 个类别的 logits

    def forward(self, x):
        # x: [B, T]
        B, T = x.shape
        # Add Positional Encoding
        emb = self.embedding(x) + self.pos_encoder[:, :T, :]
        feat = self.transformer(emb)
        logits = self.head(feat)  # [B, T, K]
        return logits, feat


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
import numpy as np
from typing import Dict, Any, Optional

# 尝试导入 Transformers，如果不存在则禁用 Teacher
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: 'transformers' library not found. LLM Teacher will be disabled.")


# ==============================================================================
# Part 1: Helper Functions (Padding & Loss)
# ==============================================================================

def _pad_to_multiple(x: torch.Tensor, multiple: int, pad_value: float = 0.0):
    """
    将时间序列 Pad 到 multiple 的倍数
    Input: [B, L, C]
    Output: [B, L_pad, C], L_orig
    """
    B, L, C = x.shape
    L_pad = ((L + multiple - 1) // multiple) * multiple
    if L_pad == L:
        return x, L
    pad_len = L_pad - L
    pad = x.new_full((B, pad_len, C), pad_value)
    return torch.cat([x, pad], dim=1), L


# ==============================================================================
# Part 2: Teacher 1 - Soft Llama (LLM)
# ==============================================================================
class SoftLlamaTeacher(nn.Module):
    def __init__(self, llm_path, adapter_path,codebook_weights, mask_token_id, device):
        super().__init__()
        self.device = device
        self.mask_token_id = mask_token_id

        print(f"[Teacher] Loading Llama from {llm_path}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, torch_dtype=torch.float16, trust_remote_code=True
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        for p in self.llm.parameters(): p.requires_grad = False

        self.llm_dim = self.llm.config.hidden_size
        self.register_buffer("codebook", codebook_weights.to(device))
        self.num_vq_codes = self.codebook.shape[0]
        self.vq_dim = self.codebook.shape[1]

        self.projector = nn.Linear(self.vq_dim, self.llm_dim).to(device)
        self.mask_embed_llama = nn.Parameter(torch.randn(1, 1, self.llm_dim).to(device))
        self.output_head = nn.Linear(self.llm_dim, self.num_vq_codes).to(device)

        self.system_prompt = "Analyze sensor sequence:"
        self.prompt_input_ids = self.tokenizer(self.system_prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            self.prompt_embeds = self.llm.get_input_embeddings()(self.prompt_input_ids)
        # === 新增：加载对齐后的权重 ===
        if adapter_path is not None and os.path.exists(adapter_path):
            print(f"[Teacher] Loading Aligned Adapters from {adapter_path}")
            checkpoint = torch.load(adapter_path, map_location=device)

            # 加载 projector
            self.projector.load_state_dict(checkpoint['projector'])

            # 加载 output_head
            self.output_head.load_state_dict(checkpoint['output_head'])

            # 极其重要：加载后冻结它们！
            # 在蒸馏阶段，Teacher 应该是全冻结的（包括 Adapter）
            # 除非你想在蒸馏时继续微调 Teacher (通常不建议)
            for p in self.projector.parameters(): p.requires_grad = False
            for p in self.output_head.parameters(): p.requires_grad = False
        else:
            print("[Teacher] WARNING: No adapter weights loaded! Initializing randomly (BAD for Distillation).")

    def forward(self, masked_ids):
        B, T = masked_ids.shape
        is_mask = (masked_ids == self.mask_token_id)

        safe_ids = masked_ids.clone()
        safe_ids[is_mask] = 0
        safe_ids = safe_ids.clamp(0, self.num_vq_codes - 1)

        vq_embeds = F.embedding(safe_ids, self.codebook)
        sensor_embeds = self.projector(vq_embeds)

        expanded_mask = self.mask_embed_llama.expand(B, T, -1)
        mask_broadcast = is_mask.unsqueeze(-1).float()
        final_sensor_embeds = sensor_embeds * (1 - mask_broadcast) + expanded_mask * mask_broadcast

        batch_prompt = self.prompt_embeds.expand(B, -1, -1)
        inputs_embeds = torch.cat([batch_prompt, final_sensor_embeds], dim=1).to(self.llm.dtype)

        outputs = self.llm(inputs_embeds=inputs_embeds, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]

        L_text = batch_prompt.shape[1]
        sensor_features = last_hidden[:, L_text:, :]

        logits = self.output_head(sensor_features.to(torch.float32))
        probs = F.softmax(logits, dim=-1)

        class TeacherOut:
            def __init__(self, probs): self.probs = probs

        return TeacherOut(probs)


# ==============================================================================
# Part 3: Student Components (Conv Patch Embedding + Transformer)
# ==============================================================================
class PatchEmbeddingConv(nn.Module):
    """
    使用卷积提取局部时序特征，并保证输出长度对齐
    """

    def __init__(self, seq_len_pad: int, patch_len: int, in_channels: int, embed_dim: int):
        super().__init__()
        # 这里的 seq_len_pad 应该是已经 Pad 过的总长度
        self.seq_len = seq_len_pad
        self.patch_len = patch_len
        self.num_patches = seq_len_pad // patch_len
        self.embed_dim = embed_dim

        # Conv stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim // 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )

        # Patch projection
        self.proj = nn.Linear(patch_len * embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, mask_bool: torch.Tensor = None) -> torch.Tensor:
        # x: [B, L_pad, C]
        B, L, C = x.shape

        # 1. Conv Stem
        h = self.stem(x.transpose(1, 2)).transpose(1, 2)  # [B, L, D]

        # 2. Patchify
        # [B, L, D] -> [B, P, patch_len, D]
        h = torch.reshape(h,(B, self.num_patches, self.patch_len, self.embed_dim))
        # Flatten patches: [B, P, patch_len*D]

        h = torch.reshape(h,(B, self.num_patches, self.patch_len * self.embed_dim))

        # 3. Project
        h = self.proj(h)  # [B, P, D]
        h = self.norm(h)
        h = h + self.pos_embed

        # 4. Masking
        if mask_bool is not None:
            w = mask_bool.unsqueeze(-1).type_as(h)
            mask_tokens = self.mask_token.expand(B, self.num_patches, self.embed_dim)
            h = h * (1 - w) + mask_tokens * w

        return h


class StrongStudent(nn.Module):
    def __init__(self, seq_len_pad, patch_len, in_channels, dim_model, num_vq_codes, nhead=4, num_layers=4):
        super().__init__()
        self.patch_embed = PatchEmbeddingConv(
            seq_len_pad=seq_len_pad,
            patch_len=patch_len,
            in_channels=in_channels,
            embed_dim=dim_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model, nhead=nhead, dim_feedforward=dim_model * 4,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.vocab_head = nn.Linear(dim_model, dim_model)

    def forward(self, x, mask_bool=None):
        emb = self.patch_embed(x, mask_bool)
        feat = self.transformer(emb)
        logits = self.vocab_head(feat)
        return logits, feat


# ==============================================================================
# Part 4: Main Model (Fixed Logic)
# ==============================================================================
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.stage = int(getattr(args, "stage", 1))
        self.device = args.device

        print(f"[Model] Init in Stage: {self.stage}")

        # 1. 解析 Stage
        self.stage = int(getattr(args, "stage", 1))
        print(f"init Model in Stage: {self.stage}")
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
        self.device = args.device
        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 15)))
        self.num_class = int(self.ds_cfg.get("num_labels", getattr(args, "num_class", 12)))

        self.seq_len=self.args.seq_len
        self.patch_len=self.args.patch_len

        self.dim_student = int(getattr(args, "dim_student", 256))

        self.student = StrongStudent(
            seq_len_pad=self.seq_len,  # Pass Padded Length
            patch_len=self.patch_len,
            in_channels=self.C,
            dim_model=self.dim_student,
            num_vq_codes=self.dim_student,
            nhead=4,
            num_layers=4
        ).to(self.device)



        # --- 5. Classifier (Stage 2) ---
        self.classifier = nn.Sequential(
            nn.Linear(self.dim_student, self.dim_student),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.dim_student, self.num_class)
        )


    def forward(self, x_imu, padding_mask=None, mode=None, labels=None):

        if not torch.is_tensor(x_imu): x_imu = torch.as_tensor(x_imu)
        B, L_orig, C = x_imu.shape
        # x_pad, L_pad = _pad_to_multiple(x_imu, self.stride, pad_value=0.0)
        x_pad = x_imu
        valid_patches = L_orig // self.patch_len
        # Student Forward
        _, feat = self.student(x_pad, mask_bool=None)  # [B, P, D]

        feat_valid = feat[:, :valid_patches, :]

        # Global Pooling
        global_feat = feat_valid.mean(dim=1)  # [B, D]

        logits = self.classifier(global_feat)  # [B, num_class]
        return logits

    def save_wrapper(self, path):
        sd = self.state_dict()
        to_save = {}
        for k, v in sd.items():
            if "vq_net" in k or "teacher" in k:
                continue
            to_save[k] = v
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(to_save, path)
        print(f"Saved model to {path}")

    def load_wrapper(self, path, map_location="cpu"):
        sd = torch.load(path, map_location=map_location)
        sd = {k: v for k, v in sd.items() if "vq_net" not in k and "teacher" not in k}
        self.load_state_dict(sd, strict=False)
        print("Loaded Student weights.")
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
    configs.ts_backbone_yaml = r"D:\fuy\MyCode\SensorLLMLib_v2\configs\ts_backbone.yaml"
    configs.debug_fake_llm = True

    configs.stage=2
    if torch.cuda.is_available() and configs.use_gpu:
        configs.device = torch.device('cuda:{}'.format(configs.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            configs.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            configs.device = torch.device("cpu")
        print('Using cpu or mps')
    model = Model(configs).to("cuda:0")
    x = torch.rand(32,96,15)
    c = model(x.to("cuda:0"),None,None)
    d = 'end'