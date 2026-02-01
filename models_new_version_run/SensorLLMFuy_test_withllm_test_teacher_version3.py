# models/SensorLLMResampler.py  (Route A)
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from models_new_version_run.VQ_VAE import IMU_VQ_Model

'''
改进：2026年1月19日

这是一个非常深入的技术问题。我会先回答你关于 **UbiPhysio** 论文逻辑的疑问，然后针对你提供的**教师-学生蒸馏架构（Teacher-Student Distillation）**代码给出改进建议。

### 一、 回答你的疑问：UbiPhysio 是怎么做的？

**1. UbiPhysio 的做法（Discrete Seq2Seq）：**
UbiPhysio 确实是先通过 VQ-VAE 得到离散词表（Codebook Indices），然后**把这些索引当成“单词”**。

* 它们将索引  直接放入 Prompt 字符串中（例如：`"tokens: 12 45 8 99..."`）。
* 然后输入到一个标准的 **Seq2Seq LLM**（如 T5 或 LLaMA）进行微调。
* **本质：** 它走的是 **“离散映射”** 路线。LLM 并没有在向量层面与 VQ 空间对齐，而是通过微调学习了“数字序列  代表某种动作描述”的映射关系。

**2. 你的做法（Feature Alignment / Distillation）：**
你的代码走的是 **“特征对齐/蒸馏”** 路线。

* 你通过 `projector` 将 VQ 向量映射到 LLM 的隐藏层空间。
* 你的 Teacher 输出的是 `semantic_features`（连续向量），让 Student 去逼近这些向量。
* **评价：** 你的方案在学术上更接近 **BEiT** 或 **Distill-BERT** 的逻辑。相比 UbiPhysio，这种做法理论上能保留更细粒度的传感器语义，因为你是在 **连续向量空间** 蒸馏知识，而不是仅仅在离散符号上做翻译。

---

### 二、 代码改进方案建议

你的代码结构已经非常完整（考虑了 Stride 对齐、Padding Mask 和投影层），但在“蒸馏效能”和“传感器特性”上还有以下改进空间：

#### 1. 改进掩码策略：从“随机掩码”到“块掩码 (Block Masking)”

传感器数据（IMU）在时间上具有极强的连续性。

* **当前问题：** `random_masking` (0.4) 会随机挖掉单个点。由于邻近点非常相似，Student 模型很容易通过插值“偷懒”猜出答案，而学不到高层语义。
* **建议：** 使用 **Block Masking**（连续挖掉一小段，例如连续 5-10 个 Patch）。这迫使模型根据上下文的运动趋势来恢复语义，能显著提升表征能力。

#### 2. 增强 Teacher 的“上下文”能力

* **当前问题：** 你的 `SoftLlamaTeacher` 的 `forward` 仅仅跑了一个 `projector(vq_embeds)`。
```python
semantic_features = self.projector(vq_embeds) 

```


这只是**静态特征转换**。它并没有发挥 LLM 的 **Transformer 序列建模** 能力。
* **建议：** 既然你加载了整个 Llama 模型，应该让数据流过 Llama 的前几层。
* **做法：** 将 `semantic_features` 拼上 `system_prompt`，输入 LLM 跑 1-2 层（Frozen），取其 Hidden States 作为蒸馏目标。这样 Student 学到的是“经过 LLM 上下文处理后”的语义，而不仅仅是单点的向量对齐。



#### 3. 引入“温度因子 (Temperature)”与特征归一化

* **当前问题：** Student 特征和 Teacher 特征（LLM 空间）的量级（Magnitude）可能完全不同。直接计算 MSE 会导致 Loss 极难收敛或梯度爆炸。
* **建议：** 在计算 `loss_distill` 之前，对两个特征进行 **LayerNorm** 或使用 **余弦相似度损失 (Cosine Similarity Loss)** 代替 MSE。
```python
# 改进方案
loss_distill = 1 - F.cosine_similarity(student_mask_feat, teacher_mask_feat, dim=-1).mean()

```



#### 4. 分类器阶段 (Stage 2) 的 Pooling 策略

* **当前问题：** 你使用了 `global_feat = feat_valid.mean(dim=1)`。
* **建议：** 对于动作识别，`Mean Pooling` 会抹平运动的时间特征。
* **方案 A：** 在 Stage 1 引入一个 `[CLS]` Token，Stage 2 只取 `[CLS]` 的输出。
* **方案 B：** 使用 `Attention Pooling` (通过一个可学习的 Query 聚合所有 Patch 的信息)。



#### 5. 增加“对比学习”辅助 Loss (Optional)

* **建议：** 在 Stage 1，除了 MSE 蒸馏，可以加一个简单的 **Contrastive Loss (如 InfoNCE)**。让 Student 模型确保“当前 Patch 的特征”与“Teacher 对应的 Patch 特征”最近，而与其他 Patch 远。这能极大增强特征的辨别力。

---

### 三、 总结建议的修改点 (代码逻辑)

```python
# 建议在 Teacher forward 中增加上下文感知
def forward(self, gt_ids):
    safe_ids = gt_ids.clamp(0, self.num_vq_codes - 1)
    vq_embeds = F.embedding(safe_ids, self.codebook) 
    soft_embeds = self.projector(vq_embeds) # [B, T, llm_dim]

    # 改进：如果显存允许，通过 LLM 跑一圈获得上下文语义
    # combined_embeds = torch.cat([self.prompt_embeds.expand(B,-1,-1), soft_embeds], dim=1)
    # with torch.no_grad():
    #     outputs = self.llm(inputs_embeds=combined_embeds, output_hidden_states=True)
    #     # 取最后一层 Hidden State 中对应 Sensor 的部分
    #     semantic_features = outputs.hidden_states[-1][:, self.prompt_input_ids.shape[1]:] 

    return soft_embeds # 或者返回增强后的 semantic_features

```

**你的方案相比 UbiPhysio 的优势：**
你的架构更像是一个 **“传感器语言模型”的预训练过程**。如果你能完成 `Student -> Projector -> LLM` 的闭环，你的模型不仅能做分类（Stage 2），未来甚至可以像 GPT 插件一样，直接插在 LLM 上，让 LLM 直接“看见”原始传感器数据。

你的逻辑是非常前沿的（基于特征蒸馏的传感器表征学习），建议重点优化 **Masking 策略** 和 **特征对齐的 Loss 函数**。



'''

import math
# 禁用 fused attention，避免 _efficient_attention_backward invalid argument
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
try:
    import yaml
except Exception:
    yaml = None

import torch
def entropy_from_probs(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # p: [B, K]
    return -(p * (p + eps).log()).sum(dim=-1)

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
    logits: torch.Tensor      # [B, Kclass]
    probs: torch.Tensor       # [B, Kclass]
    feat: torch.Tensor        # [B, D]
    reason_text: Optional[List[str]] = None   # len B, optional

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
from typing import Dict, Any


class DeepConvLSTMAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # ================== 1. 你的初始化逻辑 (保留) ==================
        print(f"[DeepConvLSTMAttnModel] Init...")
        self.stage = int(getattr(args, "stage", 1))
        # self.device = args.device # 建议在 forward 中使用 x.device，更灵活

        # -------- dataset cfg load --------
        self.dataset_key = str(getattr(args, "dataset_key", getattr(args, "data", "mhealth"))).lower()
        self.ds_cfg: Dict[str, Any] = {}
        if hasattr(args, "ds_cfg") and isinstance(args.ds_cfg, dict):
            self.ds_cfg = args.ds_cfg
        else:
            ts_yaml = getattr(args, "ts_backbone_yaml", None)
            if ts_yaml is not None:
                if not os.path.exists(ts_yaml):
                    raise FileNotFoundError(ts_yaml)
                with open(ts_yaml, "r", encoding="utf-8") as f:
                    cfg_all = yaml.safe_load(f)
                if self.dataset_key not in cfg_all:
                    raise KeyError(f"{self.dataset_key} not in {ts_yaml}")
                self.ds_cfg = cfg_all[self.dataset_key]

        # 读取参数
        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 15)))  # 传感器通道数 (NB_SENSOR_CHANNELS)
        self.num_class = int(self.ds_cfg.get("num_labels", getattr(args, "num_class", 12)))  # 类别数
        self.seq_len_orig = int(getattr(args, "seq_len", 200))

        # ================== 2. DeepConvLSTM 超参数 ==================
        # 你可以将这些也放入 configs，这里使用原代码的默认值作为 fallback
        self.num_filters = int(getattr(args, "num_filters", 64))
        self.filter_size = int(getattr(args, "filter_size", 5))
        self.num_units_lstm = int(getattr(args, "num_units_lstm", 128))
        self.num_layers_lstm = int(getattr(args, "num_layers_lstm", 2))
        # Dropout
        self.dropout_val = float(getattr(args, "dropout", 0.5))
        self.attention_dropout_val = float(getattr(args, "attention_dropout", 0.5))

        # ================== 3. 网络架构定义 (源自 main_script.py) ==================
        # 卷积层: input shape (Batch, 1, SeqLen, Channels)
        # 注意: 这里的 Conv2d 卷积核是 (filter_size, 1)，意味着它在时间维上卷积，传感器维度保持独立卷积
        self.conv2DLayer1 = nn.Conv2d(1, self.num_filters, (self.filter_size, 1), stride=(1, 1))
        self.relu1 = nn.ReLU()
        self.conv2DLayer2 = nn.Conv2d(self.num_filters, self.num_filters, (self.filter_size, 1), stride=(1, 1))
        self.relu2 = nn.ReLU()
        self.conv2DLayer3 = nn.Conv2d(self.num_filters, self.num_filters, (self.filter_size, 1), stride=(1, 1))
        self.relu3 = nn.ReLU()
        self.conv2DLayer4 = nn.Conv2d(self.num_filters, self.num_filters, (self.filter_size, 1), stride=(1, 1))
        self.relu4 = nn.ReLU()

        # LSTM层
        # input_size calculation:
        # 卷积后的输出是 (Batch, Filters, SeqLen, Sensors)。
        # 转换到 LSTM 时，会将 Filters 和 Sensors 展平。
        # 因此 LSTM input_size = num_filters * num_sensors
        lstm_input_size = self.num_filters * self.C
        self.lstm = nn.LSTM(lstm_input_size, self.num_units_lstm, self.num_layers_lstm,
                            bidirectional=False, dropout=self.dropout_val)

        self.dropout = nn.Dropout(self.dropout_val)
        self.attention_dropout = nn.Dropout(self.attention_dropout_val)

        # Attention 层 (main_script.py 特有)
        hidden_dim = self.num_units_lstm  # 单向 LSTM
        self.attentionLayer1 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh1 = nn.Tanh()
        self.attentionLayer2 = nn.Linear(hidden_dim, 1)
        self.softmax_attention = nn.Softmax(dim=0)

        # 全连接层
        self.dense_layer = nn.Linear(hidden_dim, self.num_class)

    def initHidden(self, batch_size, device):
        # 原代码逻辑：使用随机噪声初始化隐层状态
        h0 = torch.randn(self.num_layers_lstm, batch_size, self.num_units_lstm).to(device) * 0.08
        c0 = torch.randn(self.num_layers_lstm, batch_size, self.num_units_lstm).to(device) * 0.08
        return (h0, c0)

    def forward(self, x, padding_mask=None, mode=None, labels=None):
        # x shape 假设为: (Batch, SeqLen, Channels)
        # 原模型需要: (Batch, 1, SeqLen, Channels)
        # 调整维度以适配 DeepConvLSTM 的 Conv2d 输入
        if x.dim() == 3:
            x = x.unsqueeze(1)  # -> (Batch, 1, SeqLen, Channels)

        # 1. 卷积部分
        convout1 = self.relu1(self.conv2DLayer1(x))
        convout2 = self.relu2(self.conv2DLayer2(convout1))
        convout3 = self.relu3(self.conv2DLayer3(convout2))
        convout4 = self.relu4(self.conv2DLayer4(convout3))

        # 2. 变换维度适配 LSTM
        # 当前 shape: (Batch, Filters, NewSeqLen, Channels)
        # 目标 shape: (NewSeqLen, Batch, Filters * Channels)
        lstm_input = convout4.permute(2, 0, 1, 3)  # -> (NewSeqLen, Batch, Filters, Channels)
        seq_len, batch_size, _, _ = lstm_input.size()
        lstm_input = lstm_input.contiguous().view(seq_len, batch_size, -1)

        lstm_input = self.dropout(lstm_input)

        # 3. LSTM 部分
        output, hidden = self.lstm(lstm_input, self.initHidden(batch_size, x.device))
        # output shape: (SeqLen, Batch, Hidden)

        # 4. Attention 部分 (核心差异)
        past_context = output[:-1]  # 过去的所有时间步
        current = output[-1]  # 当前时间步 (最后一个)

        # 计算 Attention score
        attn_out = self.attentionLayer1(past_context)
        attn_out = self.tanh1(attn_out)
        attn_out = self.attention_dropout(attn_out)
        attn_out = self.attentionLayer2(attn_out)  # -> (SeqLen-1, Batch, 1)

        # 计算 Attention weights
        attn_weights = self.softmax_attention(attn_out)  # 在时间维度 dim=0 做 softmax

        # 加权求和
        new_context_vector = torch.sum(attn_weights * past_context, dim=0)  # -> (Batch, Hidden)

        # Skip connection: Attention结果 + 当前状态
        new_context_vector = new_context_vector + current

        # 5. 分类
        logits = self.dense_layer(new_context_vector)
        return logits,new_context_vector

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
    def __init__(self, llm_path, codebook_weights, device):
        super().__init__()
        self.device = device

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, torch_dtype=torch.float16, trust_remote_code=True
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        for p in self.llm.parameters():
            p.requires_grad = False

        self.llm_dim = self.llm.config.hidden_size

        self.register_buffer("codebook", codebook_weights.to(device))  # [K, Dvq]
        self.num_vq_codes = self.codebook.shape[0]
        self.vq_dim = self.codebook.shape[1]

        # 这两个是 teacher 唯一需要训练的部分（Stage1a）
        self.projector = nn.Linear(self.vq_dim, self.llm_dim).to(device)
        self.output_head = nn.Linear(self.llm_dim, self.num_vq_codes).to(device)

        self.system_prompt = "Analyze the following sensor sequence and predict the underlying pattern:"
        prompt_ids = self.tokenizer(self.system_prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            self.prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)  # [1, Lp, D]

    def forward(self, ids):  # ids: [B,P] 取值 [0..K-1]
        B, P = ids.shape
        ids = ids.clamp(0, self.num_vq_codes - 1)

        vq_embeds = F.embedding(ids, self.codebook)        # [B,P,Dvq]
        sensor_embeds = self.projector(vq_embeds)          # [B,P,D]

        batch_prompt = self.prompt_embeds.expand(B, -1, -1)
        inputs_embeds = torch.cat([batch_prompt, sensor_embeds], dim=1).to(self.llm.dtype)

        outputs = self.llm(inputs_embeds=inputs_embeds, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]            # [B,Lp+P,D]

        Lp = batch_prompt.shape[1]
        sensor_hidden = last_hidden[:, Lp:, :]             # [B,P,D]
        logits = self.output_head(sensor_hidden.float())   # [B,P,K]

        # logits[:, t] 自然对应 next-token：p(z_{t+1} | z_{<=t})
        return logits





# ----------------------------
# Teacher Adapter (Mean/Std -> K soft tokens in LLM embedding space)
# ----------------------------
class SoftTokenAdapter(nn.Module):
    def __init__(self, in_dim: int, llm_embed_dim: int, num_soft_tokens: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.llm_embed_dim = llm_embed_dim
        self.num_soft_tokens = num_soft_tokens

        # PH-LLM-like MLP: in -> 1024 -> 4096 -> 1024 -> (num_soft_tokens * D)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_soft_tokens * llm_embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        B = x.shape[0]
        out = self.net(x)  # [B, K*D]
        out = out.view(B, self.num_soft_tokens, self.llm_embed_dim)  # [B, Ksoft, D]
        return out


# ----------------------------
# Teacher = Frozen LLM + Adapter + Closed-set label scoring
# ----------------------------

class TeacherClassifier(nn.Module):
    """
    Closed-set classification with a frozen causal LM:
      - Build prompt text from numeric summary
      - Prepend learned soft tokens (adapter output) as prefix embeddings
      - Score each candidate label by summing log-prob of its token(s) conditioned on prefix+prompt
    """
    def __init__(
        self,
        ds_cfg: Dict[str, Any],
        llm_path: str,
        feat_dim: int,
        num_classes: int,
        num_soft_tokens: int = 4,
        label_strings: Optional[List[str]] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        self.ds_cfg = ds_cfg

        # build maps
        self.label_strings, self.id2name, self.letter2id, self.id2letter = _build_label_maps_from_ds_cfg(self.ds_cfg)

        self.num_classes = int(self.ds_cfg.get("num_labels", len(self.label_strings)))




        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        # distilgpt2 has no pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(llm_path)
        self.llm.to(device).eval()
        for p in self.llm.parameters():
            p.requires_grad = False

        self.embed = self.llm.get_input_embeddings()
        self.llm_embed_dim = self.embed.embedding_dim

        self.adapter = SoftTokenAdapter(feat_dim * 2, self.llm_embed_dim, num_soft_tokens=num_soft_tokens).to(device)

        # label_strings 就用 A-L（用于 score_labels 的候选集合）
        # 你原来那段 label_strings None -> A,B,C... 可以删掉或直接覆盖：
        self.label_strings = self.label_strings[:self.num_classes]

        # Pre-tokenize label ids (can be multi-token)
        self.label_token_ids: List[torch.Tensor] = []


        self.output_head = nn.Linear(self.llm_embed_dim, self.num_classes).to(device)
        # self.c = nn.Flatten(-1)
        # self.output_head = nn.Sequential(
        #     nn.Flatten(-1),
        #     nn.Linear(self.llm_embed_dim * num_soft_tokens, self.num_classes)
        # ).to(device)

        for s in self.label_strings:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 0:
                raise ValueError(f"Label string {s} tokenized to empty. Choose another label string.")
            self.label_token_ids.append(torch.tensor(ids, dtype=torch.long))

    @torch.no_grad()
    def generate_reason_new(
            self,
            x: torch.Tensor,  # [B, feat_dim] ✅ 新增
            soft_tokens: torch.Tensor,  # [B, Ksoft, D]
            pred_ids: torch.Tensor,  # [B] class id in [0..K-1]
            max_new_tokens: int = 48,
    ) -> List[str]:
        """
        Generate explanation text (NOT used for training).
        Conditioning: soft prefix + explain_prompt (built from x_row + predicted class id).
        """
        B = x.shape[0]
        reasons: List[str] = []

        for i in range(B):
            pred_id = int(pred_ids[i].item())
            expl_prompt = self.build_explain_prompt(x_row=x[i], pred_id=pred_id)

            enc = self.tokenizer(expl_prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attn_mask = enc["attention_mask"].to(self.device)

            # prompt embeddings
            prompt_emb = self.embed(input_ids)  # [1, L, D]

            # concat soft prefix
            inputs_embeds = torch.cat([soft_tokens[i:i + 1], prompt_emb], dim=1)  # [1, Ksoft+L, D]
            prefix_mask = torch.ones((1, soft_tokens.shape[1]), dtype=attn_mask.dtype, device=self.device)
            full_mask = torch.cat([prefix_mask, attn_mask], dim=1)

            gen_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,          # ✅ 给一点随机性，避免固定套话循环
                temperature=0.7,
                top_p=0.9,
                num_beams=1,
                repetition_penalty=1.15, # ✅ 抑制重复
                no_repeat_ngram_size=4,  # ✅ 禁止4-gram重复
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )


            text_all = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            # 解析：优先提取 "Reason:" 之后的内容；否则返回全文
            if "Reason:" in text_all:
                reason = text_all.split("Reason:")[-1].strip()
            else:
                reason = text_all.strip()

            reasons.append(reason[:400])

        return reasons

    @torch.no_grad()
    def generate_reason(
            self,
            soft_tokens: torch.Tensor,  # [B, Ksoft, D]
            prompt_texts: List[str],  # len B (same as in build_prompt)
            pred_ids: torch.Tensor,  # [B] predicted class index
            max_new_tokens: int = 48,
    ) -> List[str]:
        """
        Generate short explanation text conditioned on:
          soft prefix tokens + prompt + 'Class: X\\nReason:'
        This is NOT used for training.
        """
        B = len(prompt_texts)
        reasons: List[str] = []

        # We generate per-sample to keep logic simple & avoid padding edge cases
        for i in range(B):
            cls = self.label_strings[int(pred_ids[i].item())]

            # Build an explanation prompt (grounded, short, avoid medical advice style)
            # You can switch to a more "why-class" style if you prefer.
            expl_prompt = (
                    prompt_texts[i]
                    + f" {cls}\n"
                    + "Reason: Explain briefly using the provided summary values only "
                      "(mention 1-3 key stats; avoid assumptions).\n"
                    + "Reason:"
            )

            enc = self.tokenizer(expl_prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attn_mask = enc["attention_mask"].to(self.device)

            # Embeddings for prompt + soft prefix
            prompt_emb = self.embed(input_ids)  # [1, L, D]
            inputs_embeds = torch.cat([soft_tokens[i:i + 1], prompt_emb], dim=1)  # [1, Ksoft+L, D]
            prefix_mask = torch.ones((1, soft_tokens.shape[1]), dtype=attn_mask.dtype, device=self.device)
            full_mask = torch.cat([prefix_mask, attn_mask], dim=1)

            gen_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic (more reproducible)
                num_beams=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # Decode only the newly generated part.
            # Because we used inputs_embeds, gen_ids includes generated token ids only for continuation.
            # But some HF versions may include full sequence ids; safe approach: decode whole and strip prompt.
            text_all = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            # Best-effort: extract substring after the last "Reason:"
            if "Reason:" in text_all:
                reason = text_all.split("Reason:")[-1].strip()
            else:
                reason = text_all.strip()

            # Keep it short (avoid runaway)
            reasons.append(reason[:400])

        return reasons

    @torch.no_grad()
    def build_explain_prompt(self, x_row: torch.Tensor, pred_id: int) -> str:
        sr = self.ds_cfg.get("sample_rate", None)
        ch = self.ds_cfg.get("channel_num", None)
        setup = []
        if ch is not None: setup.append(f"{int(ch)}-channel IMU")
        if sr is not None: setup.append(f"{int(sr)}Hz")
        setup_str = ", ".join(setup) if setup else "IMU"

        D = x_row.shape[0]
        vals = x_row[: min(12, D)].tolist()
        vals_str = ", ".join([f"{v:.3f}" for v in vals])

        letter = self.id2letter[pred_id]
        name = self.id2name[pred_id]

        return (
            "## Instruction: You are an expert in IMU-based HAR.\n"
            f"## Sensor setup: {setup_str}.\n"
            "## Input: statistical summaries (mean/variance) of IMU signals.\n"
            f"Summary(first_dims): [{vals_str}]\n"
            f"Predicted Class: {letter} ({pred_id}): {name}\n"
            "Write exactly TWO lines:\n"
            "Analysis: mention at least TWO numeric values from Summary(first_dims) and describe what they suggest.\n"
            "Reason: justify the predicted class in one sentence, referencing at least ONE numeric value.\n"
            "Do NOT repeat any sentence.\n"
            "Analysis:"
        )

    def build_prompt(self, x: torch.Tensor) -> List[str]:
        B, D = x.shape
        sr = self.ds_cfg.get("sample_rate", None)
        ch = self.ds_cfg.get("channel_num", None)

        letters = self.label_strings[:self.num_classes]
        letter_list = ", ".join(letters)

        # category block
        lines = []
        for i in range(self.num_classes):
            lines.append(f"{self.id2letter[i]} ({i}): {self.id2name[i]}")
        cat_block = "\n".join(lines)

        prompts = []
        for i in range(B):
            vals = x[i, : min(12, D)].tolist()
            vals_str = ", ".join([f"{v:.3f}" for v in vals])

            setup = []
            if ch is not None: setup.append(f"{int(ch)}-channel IMU")
            if sr is not None: setup.append(f"{int(sr)}Hz")
            setup_str = ", ".join(setup) if setup else "IMU"
            # new prompt
            # prompts.append(
            #     "## Instruction: You are an expert in IMU-based human activity recognition (HAR).\n"
            #     f"## Sensor setup: {setup_str}.\n"
            #     "## Input: statistical summaries (mean/variance) of IMU signals.\n"
            #     f"Summary(first_dims): [{vals_str}]\n"
            #     "## Candidate actions (choose exactly one):\n"
            #     f"{cat_block}\n"
            #     f"## Output format: Class: one of {letter_list}\n"
            #     "Class:"
            # )
            prompts.append(
                "Task: classify the activity from sensor summary.\n" f"Summary(first_dims): [{vals_str}]\n" "Answer:")
        return prompts





    def score_labels(
        self,
        soft_tokens: torch.Tensor,     # [B, Ksoft, D]
        prompt_texts: List[str],       # len B
        detach_llm: bool = False,  # True=推理省显存/不训adapter；False=训练adapter
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          scores: [B, Kclass]  (log-likelihood scores)
          feat:   [B, D]       (teacher feature for distill: use hidden state at last prompt position)
        """
        ctx = torch.no_grad() if detach_llm else torch.enable_grad()
        with ctx:
            # print("grad_enabled:", torch.is_grad_enabled(), "inference_mode:", torch.is_inference_mode_enabled())
            # print("soft_tokens.requires_grad:", soft_tokens.requires_grad)
            # print("adapter any grad param:", any(p.requires_grad for p in self.adapter.parameters()))

            B = len(prompt_texts)
            K = self.num_classes

            # Tokenize prompts
            enc = self.tokenizer(
                prompt_texts, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = enc["input_ids"].to(self.device)          # [B, L]
            attn_mask = enc["attention_mask"].to(self.device)     # [B, L]
            L = input_ids.shape[1]

            # Build embeddings for prompt tokens
            prompt_emb = self.embed(input_ids)                    # [B, L, D]
            # Concatenate soft prefix embeddings
            inputs_embeds = torch.cat([soft_tokens, prompt_emb], dim=1)  # [B, Ksoft+L, D]
            # Attention mask for prefix
            prefix_mask = torch.ones((B, soft_tokens.shape[1]), dtype=attn_mask.dtype, device=self.device)
            full_mask = torch.cat([prefix_mask, attn_mask], dim=1)       # [B, Ksoft+L]

            # First run: get hidden states at end of prompt (feature)
            out0 = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            # Feature: last hidden state at the last prompt token position (before label)
            # position = Ksoft + (length_of_prompt_tokens - 1) for each sample (taking padding into account)
            last_hidden = out0.hidden_states[-1]  # [B, Ksoft+L, D]
            prompt_lens = attn_mask.sum(dim=1)    # [B]
            pos = soft_tokens.shape[1] + (prompt_lens - 1)  # [B]
            feat = last_hidden[torch.arange(B, device=self.device), pos, :]  # [B, D]


            # Now compute label scores by conditioning on prefix+prompt and next tokens = label tokens.
            # We do this by expanding batch for each class and appending label tokens.
            max_label_len = max(len(t) for t in self.label_token_ids)
            #
            # Build expanded inputs for each (sample, class)
            # expanded size: B*K
            expand_BK = B * K
            soft_rep = soft_tokens.unsqueeze(1).expand(B, K, soft_tokens.shape[1], soft_tokens.shape[2]).contiguous()
            soft_rep = soft_rep.view(expand_BK, soft_tokens.shape[1], soft_tokens.shape[2])  # [BK, Ksoft, D]
            prompt_emb_rep = prompt_emb.unsqueeze(1).expand(B, K, L, self.llm_embed_dim).contiguous()
            prompt_emb_rep = prompt_emb_rep.view(expand_BK, L, self.llm_embed_dim)          # [BK, L, D]

            inputs_embeds_BK = torch.cat([soft_rep, prompt_emb_rep], dim=1)                 # [BK, Ksoft+L, D]
            full_mask_BK = full_mask.unsqueeze(1).expand(B, K, full_mask.shape[1]).contiguous()
            full_mask_BK = full_mask_BK.view(expand_BK, full_mask.shape[1])                  # [BK, Ksoft+L]

            # Create label token ids padded to max_label_len
            label_ids = torch.full((K, max_label_len), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
            label_valid = torch.zeros((K, max_label_len), dtype=torch.bool)
            for ci, ids in enumerate(self.label_token_ids):
                label_ids[ci, : len(ids)] = ids
                label_valid[ci, : len(ids)] = True
            # expand to BK
            label_ids_BK = label_ids.unsqueeze(0).expand(B, K, max_label_len).contiguous().view(expand_BK, max_label_len)
            label_valid_BK = label_valid.unsqueeze(0).expand(B, K, max_label_len).contiguous().view(expand_BK, max_label_len)

            # We need embeddings for label tokens to append
            label_emb_BK = self.embed(label_ids_BK.to(self.device))  # [BK, T, D]
            inputs_embeds_BK2 = torch.cat([inputs_embeds_BK, label_emb_BK], dim=1)  # [BK, Ksoft+L+T, D]
            # attention mask: label tokens all "present" (even pads) but we'll mask in scoring
            label_mask = torch.ones((expand_BK, max_label_len), dtype=full_mask_BK.dtype, device=self.device)
            attn_mask_BK2 = torch.cat([full_mask_BK, label_mask], dim=1)  # [BK, Ksoft+L+T]

            out = self.llm(
                inputs_embeds=inputs_embeds_BK2,
                attention_mask=attn_mask_BK2,
                use_cache=False,
            )
            logits = out.logits  # [BK, Ksoft+L+T, vocab]

            # For causal LM, token t is predicted at position t-1.
            # We want log P(label_token_j | prefix+prompt+previous label tokens)
            # The first label token is predicted at position (Ksoft+L-1).
            start = soft_tokens.shape[1] + L - 1
            # Collect logprobs at each label position
            logprobs = F.log_softmax(logits[:, start : start + max_label_len, :], dim=-1)  # [BK, T, vocab]
            gather = logprobs.gather(dim=-1, index=label_ids_BK.to(self.device).unsqueeze(-1)).squeeze(-1)  # [BK, T]
            # Mask out padded label positions
            gather = gather * label_valid_BK.to(self.device).float()
            scores_BK = gather.sum(dim=1)  # [BK]
            scores = scores_BK.view(B, K)  # [B, K]
        # prompt_lens = attn_mask.sum(dim=1)
        # pos = soft_tokens.shape[1] + (prompt_lens - 1)
        # feat = last_hidden[torch.arange(B, device=self.device), pos, :]  # [B, D]

        # 6. 映射回 VQ 空间
        # [B, L_sensor, Num_Codes]
        # featc = self.c(feat)
        # scores = self.output_head(feat.to(torch.float32))

        return scores, feat


    def forward(self, x: torch.Tensor, return_reason: bool = False,detach_llm = False) -> TeacherOut:
        x = x.to(self.device)
        soft_tokens = self.adapter(x)  # [B, Ksoft, D]
        prompts = self.build_prompt(x)
        detach_llm = False # True=推理省显存/不训adapter；False=训练adapter
        scores, feat = self.score_labels(soft_tokens, prompts,detach_llm=detach_llm)  # [B, K], [B, D]
        probs = F.softmax(scores, dim=-1)

        reason_text = None
        if return_reason:
            pred_ids = scores.argmax(dim=-1)  # [B] 这是“数字类ID”(0..K-1)，不是字母
            reason_text = self.generate_reason(
                x=x,  # ✅ 新增：传原始统计量
                soft_tokens=soft_tokens,
                pred_ids=pred_ids,
                max_new_tokens=48,
            )
        return TeacherOut(logits=scores, probs=probs, feat=feat, reason_text=reason_text)


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
        self.vocab_head = nn.Linear(dim_model, num_vq_codes)

    def forward(self, x, mask_bool=None):
        emb = self.patch_embed(x, mask_bool)
        feat = self.transformer(emb)
        logits = self.vocab_head(feat)
        return logits, feat

# ----------------------------
# Loss for Stage2 (KD modes A/B/C)
# ----------------------------
def kd_loss(
    student_logits: torch.Tensor,     # [B, K]
    y: torch.Tensor,                 # [B]
    teacher_logits: torch.Tensor,    # [B, K]
    distill_mode: str = "A",         # "A"|"B"|"C"
    T: float = 2.0,
    lambda_kd: float = 1.0,
    teacher_probs: Optional[torch.Tensor] = None,  # [B, K]
    student_feat: Optional[torch.Tensor] = None,   # [B, d]
    teacher_feat: Optional[torch.Tensor] = None,   # [B, D]
    feat_proj: Optional[nn.Module] = None,
    lambda_feat: float = 0.2,
    conf_tau: float = 0.6,
) -> Tuple[torch.Tensor, dict]:
    """
    A: CE + KL
    B: CE + w * KL  (w from entropy or max prob)
    C: CE + w * KL + feature distill
    """
    Bsz, K = student_logits.shape
    L_ce = F.cross_entropy(student_logits, y)

    # KD term
    p_t = F.softmax(teacher_logits / T, dim=-1)
    p_s = F.log_softmax(student_logits / T, dim=-1)
    L_kd = F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)

    w = torch.ones((Bsz,), device=student_logits.device)
    if distill_mode.upper() in ["B", "C"]:
        if teacher_probs is None:
            teacher_probs = F.softmax(teacher_logits, dim=-1)
        # Option: entropy-based weight
        H = entropy_from_probs(teacher_probs)              # [B]
        w_ent = 1.0 - H / math.log(K)                      # normalized to [0,1] (roughly)
        w_ent = torch.clamp(w_ent, 0.0, 1.0)
        # Option: confidence thresholding using max prob
        c = teacher_probs.max(dim=-1).values
        w_thr = (c >= conf_tau).float()
        # Combine: you can choose either; here we multiply for safety
        w = w_ent * w_thr

        # apply per-sample weighting to KD by scaling logits loss approx:
        # simplest: scale batch KD by mean weight (stable)
        L_kd = L_kd * (w.mean().detach())

    total = L_ce + lambda_kd * L_kd
    stats = {
        "L_ce": float(L_ce.detach().cpu()),
        "L_kd": float(L_kd.detach().cpu()),
        "w_mean": float(w.mean().detach().cpu()),
    }

    # Feature distill for C
    if distill_mode.upper() == "C":
        if (student_feat is not None) and (teacher_feat is not None) and (feat_proj is not None):
            s2t = feat_proj(student_feat)
            s2t = F.normalize(s2t, dim=-1)
            t = F.normalize(teacher_feat, dim=-1).to(s2t.device)
            L_feat = F.mse_loss(s2t, t)
            total = total + lambda_feat * L_feat
            stats["L_feat"] = float(L_feat.detach().cpu())
        else:
            stats["L_feat"] = float("nan")

    return total, stats
import re
from typing import Dict, Any, List, Tuple

def _build_label_maps_from_ds_cfg(ds_cfg: Dict[str, Any]) -> Tuple[List[str], Dict[int, str], Dict[str, int], Dict[int, str]]:
    """
    Returns:
      label_strings: ['A','B',...]
      id2name: {0: 'Walking Forward', ...}  # stripped numbering
      letter2id: {'A':0, ...}
      id2letter: {0:'A', ...}
    """
    num_labels = int(ds_cfg.get("num_labels", 0))
    if num_labels <= 0:
        # fallback: infer from id2label length
        id2label = ds_cfg.get("id2label", {})
        num_labels = len(id2label)

    # A, B, C ... (support >26 if you ever need)
    letters = []
    for i in range(num_labels):
        if i < 26:
            letters.append(chr(ord("A") + i))
        else:
            letters.append(f"CLASS{i}")  # fallback; for your 12-class it's A-L

    # Parse id2label and strip leading "1." style numbering if present
    raw = ds_cfg.get("id2label", {})
    # yaml may load keys as int or str
    id2name: Dict[int, str] = {}
    for k, v in raw.items():
        idx = int(k)
        s = str(v).strip()
        # remove leading "1.", "12.", "1)" etc.
        s = re.sub(r"^\s*\d+\s*[\.\)\-:]\s*", "", s)
        id2name[idx] = s

    # If id2label is missing/incomplete, create placeholders
    for i in range(num_labels):
        if i not in id2name:
            id2name[i] = f"Class{i}"

    letter2id = {letters[i]: i for i in range(num_labels)}
    id2letter = {i: letters[i] for i in range(num_labels)}
    return letters, id2name, letter2id, id2letter

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

        self.seq_len_orig = int(getattr(args, "seq_len", 200))  # 原始数据长度

        self.vq_net = IMU_VQ_Model(args)

        vqvae_path = getattr(args, "vqvae_path", None)
        self.qua_path = self.ds_cfg.get(vqvae_path, None)

        if self.qua_path is not None:
            vq_net_state_dict = torch.load(self.qua_path + os.sep + "best_wrapper.pth", map_location='cpu')
            if 'state_dict' in vq_net_state_dict:
                vq_net_state_dict = vq_net_state_dict['state_dict']
            self.vq_net.load_state_dict(vq_net_state_dict, strict=True)
            self.vq_net.eval()  # Set the model to evaluation mode

            # set vq net requires_grad to False
            for param in self.vq_net.parameters():
                param.requires_grad = False
            # --------------------------------------------
        # --- 2. Calculate Alignment (Critical) ---
        # VQ Stride = stride_t ^ down_t (e.g., 2^3 = 8)
        self.vq_stride = self.vq_net.stride_t ** self.vq_net.down_t
        self.patch_len = self.vq_stride  # Student Patch MUST match VQ Stride

        # Calculate Padded Length
        # e.g., if L=195, Stride=8 -> L_pad=200
        self.seq_len_pad = ((self.seq_len_orig + self.vq_stride - 1) // self.vq_stride) * self.vq_stride
        self.P = self.seq_len_pad // self.patch_len

        print(
            f"[Model Alignment] Orig={self.seq_len_orig}, Stride={self.vq_stride} -> Padded={self.seq_len_pad}, Patches(P)={self.P}")

        # --- 3. Initialize Student ---
        self.num_primitives = self.vq_net.code_num
        self.mask_token_id = self.num_primitives
        self.dim_student = int(getattr(args, "dim_student", 256))

        # self.student = StrongStudent(
        #     seq_len_pad=self.seq_len_pad,  # Pass Padded Length
        #     patch_len=self.patch_len,
        #     in_channels=self.C,
        #     dim_model=self.dim_student,
        #     num_vq_codes=self.num_primitives,
        #     nhead=4,
        #     num_layers=4
        # ).to(self.device)

        self.student = DeepConvLSTMAttention(args=args).to(self.device)

        self.mask_rate=args.mask_rate
        # --- 4. Initialize Teacher 2 (SoftLlama) ---
        self.lambda_distill = float(getattr(args, "lambda_distill", 1))
        self.teacher = None
        self.train_mode = str(getattr(args, "train_mode", "student_distill_B"))
        # 可选： "teacher_lm", "student_ce", "student_distill_B"
        self.distill_temp = float(getattr(args, "distill_temp", 2.0))
        self.ss_ratio = float(getattr(args, "ss_ratio", -0.5))  # scheduled sampling: 用 gt 的概率

        if self.stage == 1 and hasattr(args, 'llama_name'):
            # Load Codebook safely
            if hasattr(self, 'qua_path') and self.qua_path:
                cb_path = self.qua_path + os.sep + "best_codebook.pth"
                if os.path.exists(cb_path):
                    codebook_weights = torch.load(cb_path, map_location="cpu")
                else:
                    codebook_weights = self.vq_net.quantizer.codebook.data.cpu()
            else:
                codebook_weights = self.vq_net.quantizer.codebook.data.cpu()
            alignment_path = self.ds_cfg.get("alignment_path", None)
            adapter_path = alignment_path+ os.sep + "best_wrapper.pth"
            adapter_path = None
            self.teacher = TeacherClassifier(
                ds_cfg=self.ds_cfg,
                llm_path=args.llama_name,
                feat_dim=self.C,
                num_classes=self.num_class,
                # adapter_path=adapter_path,
                # codebook_weights=codebook_weights,
                # mask_token_id=self.mask_token_id,
                device=self.device
            )

        # --- 5. Classifier (Stage 2) ---
        if self.stage == 2:
            self.classifier = nn.Sequential(
                nn.Linear(self.dim_student, self.dim_student),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.dim_student, self.num_class)
            )

    def _load_vq_weights(self, args):
        vqvae_path = getattr(args, "vqvae_path", None)
        if hasattr(self, 'ds_cfg'):
            self.qua_path = self.ds_cfg.get(vqvae_path, None)
        if hasattr(self, 'qua_path') and self.qua_path:
            # Load logic specific to your checkpoint format
            # ckpt = torch.load(...)
            # self.vq_net.load_state_dict(...)
            print(f"[Model] Loading VQ Weights from {self.qua_path} (Simulated)")
            pass

    def random_masking(self, B, P, mask_ratio):
        noise = torch.rand(B, P, device=self.device)
        return noise < mask_ratio

    def forward(self, x_imu, padding_mask=None, mode=None, labels=None,mean=None,var=None):

        if not torch.is_tensor(x_imu): x_imu = torch.as_tensor(x_imu)
        B, L_orig, C = x_imu.shape


        # ===========================================================
        x_pad, L_pad = _pad_to_multiple(x_imu, self.vq_stride, pad_value=0.0)
        x_pad =x_imu

        valid_patches = L_orig // self.patch_len
        loss_valid_mask = torch.zeros((B, self.P), device=self.device, dtype=torch.bool)
        loss_valid_mask[:, :valid_patches] = True

        # ====================
        # Stage 1: Pretrain
        # ====================
        if self.stage == 1:

            assert self.teacher is not None, "Need teacher for teacher_lm mode"
            # t_logits = self.teacher(gt_ids[:, :V])[:, :V - 1, :]  # [B,V-1,K]
            # detach_llm # True=推理省显存/不训adapter；False=训练adapter


            # -------- train_mode 1: teacher_lm --------
            if self.train_mode == "teacher_lm":
                tout = self.teacher(torch.cat([mean, var], dim=-1), return_reason=False, detach_llm=False)  # [B,V-1,K]
                tlogits = tout.logits
                tprobs = tout.probs
                tfeat = tout.feat
                treason = tout.reason_text
                teacher_loss = F.cross_entropy(tlogits, labels)
                return teacher_loss,None,treason # 1.9519




            # Student forward (NO MASK) for both student_ce and student_distill_B
            student_logits, sfeat  = self.student(x_pad)  # [B,P,K]
            tout = self.teacher(torch.cat([mean, var], dim=-1), return_reason=False, detach_llm=True)  # [B,V-1,K]
            tlogits = tout.logits
            tprobs = tout.probs
            tfeat = tout.feat
            # -------- train_mode 2: student_ce --------
            feat_proj = None
            loss, stats = kd_loss(
                student_logits=student_logits,
                y=labels,
                teacher_logits=tlogits.detach(),
                distill_mode='A', #" A or B"
                T=2.0,
                lambda_kd=1.0,
                teacher_probs=tprobs.detach(),
                student_feat=sfeat,
                teacher_feat=tfeat.detach(),
                feat_proj=feat_proj,
                lambda_feat=0.2,
                conf_tau=0.6,
            )





            return loss, student_logits, stats


        # ====================
        # Stage 2: Classify
        # ====================
        elif self.stage == 2:
            # Student Forward
            logits, feat = self.student(x_pad)  # [B, P, D]

            # --- Pooling with Mask ---
            # 只对有效的 Patch 进行平均，忽略 Padding 部分
            # feat: [B, P, D] -> [B, valid_patches, D]
            # feat_valid = feat[:, :valid_patches, :]
            #
            # # Global Pooling
            # global_feat = feat_valid.mean(dim=1)  # [B, D]
            #
            # logits = self.classifier(global_feat)  # [B, num_class]
            # logits = feat
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


    def save_teacher_head(self, path):
        assert self.teacher is not None
        sd = {
            "adapter": self.teacher.adapter.state_dict(),
            "output_head": self.teacher.output_head.state_dict(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(sd, path)
        print(f"Saved teacher head to {path}")

    def load_teacher_head(self, path, map_location="cpu"):
        assert self.teacher is not None
        ckpt = torch.load(path, map_location=map_location)
        self.teacher.adapter.load_state_dict(ckpt["adapter"], strict=True)
        self.teacher.output_head.load_state_dict(ckpt["output_head"], strict=True)
        print(f"Loaded teacher head from {path}")

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