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

class TeacherOut:
    def __init__(self, probs_full, chosen_pos):
        self.probs = probs_full  # [B,P,K]
        self.chosen_pos = chosen_pos  # [B]
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
# 头部需要增加这个 import
from peft import PeftModel


# ==============================================================================
# Part 2: Teacher 1 - LoRA Llama Teacher (New & Correct)
# ==============================================================================
class SoftLlamaTeacher(nn.Module):
    def __init__(self, llm_path, adapter_path, codebook_weights, mask_token_id, device):
        super().__init__()
        self.device = device
        self.mask_token_id = mask_token_id

        # 1. 加载 Tokenizer (和训练时保持完全一致)
        print(f"[Teacher] Loading Tokenizer from {llm_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. 注册 Special Tokens (必须和 LoRA 训练时一致)
        # 假设 codebook_weights.shape[0] 是 512
        K = codebook_weights.shape[0]
        prim_tokens = [f"<P{i:03d}>" for i in range(K)] + ["<PMASK>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": prim_tokens})

        # 3. 加载 Base Model
        print(f"[Teacher] Loading Base Llama from {llm_path}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.float16,
            # device_map="auto" # 建议注释掉，手动控制 to(device) 更稳
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # 4. 加载你训练好的 LoRA Adapter
        if adapter_path and os.path.exists(adapter_path):
            print(f"[Teacher] Loading LoRA Adapter from {adapter_path}...")
            self.llm = PeftModel.from_pretrained(
                self.llm,
                adapter_path,
                torch_dtype=torch.float16
            )
            # 融合 LoRA 权重能加快推理速度 (可选)
            # self.llm = self.llm.merge_and_unload()
        else:
            print(f"[Teacher] WARNING: Adapter path {adapter_path} not found! Teacher is dumb.")

        # 5. 冻结所有参数 & 移至 GPU
        self.llm.eval().to(self.device)
        for p in self.llm.parameters():
            p.requires_grad = False

        # 缓存 Token ID 映射，加速 forward
        # 我们需要知道 <P000> 对应的 tokenizer id 是多少
        self.prim_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in prim_tokens[:-1]]  # 不含 PMASK
        self.prim_ids_tensor = torch.tensor(self.prim_ids, device=self.device)

    def forward(self, masked_ids):
        """
        masked_ids: [B, P] 包含了 VQ index，其中被 mask 的位置是 mask_token_id
        """
        B, P = masked_ids.shape

        # 1. 构造 Prompt (必须与 LoRA 训练时一致)
        # 这是一个稍微耗时的操作，但在 batch size 不大时可以接受
        batch_prompts = []

        # 把 tensor 转回 cpu list 处理字符串
        ids_list = masked_ids.cpu().tolist()

        for b in range(B):
            seq_str = []
            for t in range(P):
                val = ids_list[b][t]
                if val == self.mask_token_id:
                    seq_str.append("<PMASK>")
                else:
                    seq_str.append(f"<P{val:03d}>")

            # Prompt 模板
            prompt = (
                "Analyze the sensor sequence and fill in the mask.\n"
                f"Sequence: {' '.join(seq_str)}\n"
                "Answer:"  # 注意这里没有空格，紧接着预测下一个 token
            )
            batch_prompts.append(prompt)

        # 2. Tokenizer
        enc = self.tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # 3. LLM Inference
        with torch.no_grad():
            outputs = self.llm(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask
            )
            # 取最后一个 token 的 logits (对应 "Answer:" 后面那个词)
            # [B, Vocab_Size]
            next_token_logits = outputs.logits[:, -1, :]

        # 4. 只取 Sensor Tokens 的 logits
        # 我们只关心 <P000>~<P511> 的概率，忽略英文单词
        sensor_logits = next_token_logits[:, self.prim_ids_tensor]  # [B, K]

        # 5. Softmax 得到概率分布
        probs = F.softmax(sensor_logits, dim=-1)  # [B, K]

        class TeacherOut:
            def __init__(self, probs):
                self.probs = probs

        return TeacherOut(probs)


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class TeacherOut:
    def __init__(self, probs_full, chosen_pos):
        self.probs = probs_full          # [B,P,K]
        self.chosen_pos = chosen_pos     # [B]


class SoftLlamaTeacherLoRA(nn.Module):
    """
    LoRA Llama teacher:
      input: masked_ids [B,P] (mask positions use mask_token_id=K)
      output: probs_full [B,P,K] (only one position per sample is informative)
    """
    def __init__(self, llm_path: str, adapter_path: str, K: int, mask_token_id: int,
                 device: torch.device, max_len: int = 512, temperature: float = 0.5):
        super().__init__()
        self.device = device
        self.K = int(K)
        self.mask_token_id = int(mask_token_id)
        self.max_len = int(max_len)
        self.temperature = float(temperature)

        # ---- tokenizer: MUST match training ----
        # if you saved tokenizer in adapter dir, load it from adapter_path
        tok_src = adapter_path if (adapter_path and os.path.exists(adapter_path)) else llm_path
        print(f"[Teacher] Loading tokenizer from: {tok_src}")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=False)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        prim_tokens = [f"<P{i:03d}>" for i in range(self.K)] + ["<PMASK>"]
        # if tokenizer already has them, add_special_tokens will add 0
        self.tokenizer.add_special_tokens({"additional_special_tokens": prim_tokens})

        # cache primitive token ids (<P000>..<P{K-1}>)
        prim_ids = [self.tokenizer.convert_tokens_to_ids(f"<P{i:03d}>") for i in range(self.K)]
        if any(x < 0 for x in prim_ids):
            bad = [i for i, x in enumerate(prim_ids) if x < 0][:10]
            raise ValueError(f"[Teacher] primitive token id mapping failed. Example bad ids: {bad}")
        self.register_buffer("prim_ids_tensor", torch.tensor(prim_ids, dtype=torch.long), persistent=False)

        # ---- base model ----
        print(f"[Teacher] Loading base model from: {llm_path}")
        base = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        base.resize_token_embeddings(len(self.tokenizer))

        # ---- load LoRA adapter ----
        if adapter_path and os.path.exists(adapter_path):
            print(f"[Teacher] Loading LoRA adapter from: {adapter_path}")
            self.llm = PeftModel.from_pretrained(base, adapter_path, torch_dtype=torch.float16)
            # optional: merge for faster inference
            # self.llm = self.llm.merge_and_unload()
        else:
            raise FileNotFoundError(f"[Teacher] adapter_path not found: {adapter_path}")

        self.llm.eval().to(self.device)
        for p in self.llm.parameters():
            p.requires_grad = False

        tid = self.tokenizer.convert_tokens_to_ids("<P000>")
        assert tid != self.tokenizer.unk_token_id, "P000 is not registered as a single special token!"

        # 1) 看 tokenizer 里到底注册了多少个 <Pxxx>
        p_tokens = [t for t in self.tokenizer.get_vocab().keys() if t.startswith("<P") and t.endswith(">")]
        p_tokens = sorted(p_tokens)

        # print("Number of <Pxxx> tokens:", len(p_tokens))
        # print("Last 5 tokens:", p_tokens[-5:])
        # print("Does <P512> exist?", "<P512>" in p_tokens)
        # print("Does <PMASK> exist?", "<PMASK>" in p_tokens)












    def _build_prompt_one_mask(self, ids_row: torch.Tensor, pick_row: Optional[torch.Tensor] = None) -> tuple[str, int]:
        """
        ids_row: [P], contains mask_token_id at masked positions
        pick_row: [P] bool mask; if provided, only pick where pick_row==True AND ids_row==mask_token_id
        """
        # ---- always do selection on CPU to avoid cuda/cpu mismatch ----
        ids_row_cpu = ids_row.detach().to("cpu")
        pick_row_cpu = pick_row.detach().to("cpu") if pick_row is not None else None

        mask_pos = (ids_row_cpu == self.mask_token_id)

        if pick_row_cpu is not None:
            cand = (mask_pos & pick_row_cpu.bool()).nonzero(as_tuple=False).view(-1)
        else:
            cand = mask_pos.nonzero(as_tuple=False).view(-1)

        if cand.numel() == 0:
            return "", -1

        # randint on CPU is fine
        mpos = int(cand[torch.randint(0, cand.numel(), (1,)).item()].item())

        row_list = ids_row_cpu.tolist()
        seq = []
        for t, v in enumerate(row_list):
            if (t == mpos) or (int(v) == self.mask_token_id):
                seq.append("<PMASK>")
            else:
                seq.append(f"<P{int(v):03d}>")

        prompt = (
            "Analyze the sensor sequence and fill in the mask.\n"
            f"Sequence: {' '.join(seq)}\n"
            "Answer: "
        )
        return prompt, mpos

    @torch.no_grad()
    def forward(self, masked_ids: torch.Tensor, pick_from_mask: Optional[torch.Tensor] = None) -> TeacherOut:
        """
        masked_ids: [B,P]  (被mask的位置 = self.mask_token_id)
        pick_from_mask: [B,P] bool，可选：要求只从 (pick_from_mask==True) 的mask位置里挑一个来问LLM
        return:
          probs_full: [B,P,K] (默认 uniform，只有 chosen_pos 那个位置是 teacher 分布)
          chosen_pos: [B] (每个样本实际挑的mask位置；若无可选mask则为 -1)
        """
        B, P = masked_ids.shape
        prompts = []
        chosen_pos = torch.full((B,), -1, device=self.device, dtype=torch.long)

        # --- 逐样本构建prompt，保证 prompts 长度一定是 B ---
        ids_cpu = masked_ids.detach().to("cpu")
        pick_cpu = pick_from_mask.detach().to("cpu") if pick_from_mask is not None else None

        for b in range(B):
            ids_row = masked_ids[b]  # GPU tensor
            pick_row = pick_cpu[b] if pick_cpu is not None else None

            ptxt, mpos = self._build_prompt_one_mask(ids_row, pick_row=pick_row)
            # 若该样本没有任何可选mask位置，给一个“dummy prompt”，并保持 chosen_pos=-1
            if mpos < 0:
                ptxt = "Analyze the sensor sequence and fill in the mask.\nSequence: <PMASK>\nAnswer: "
            prompts.append(ptxt)
            # print(prompts[0])
            chosen_pos[b] = mpos

        # --- tokenize (prompts 一定非空，且长度=B) ---
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len
        ).to(self.device)

        out = self.llm(input_ids=enc.input_ids, attention_mask=enc.attention_mask)

        # 用 last non-pad 位置拿 logits（更稳）
        last_pos = enc.attention_mask.sum(dim=1) - 1  # [B]
        last_logits = out.logits[torch.arange(B, device=self.device), last_pos, :]  # [B,V]

        prim_logits = last_logits.index_select(dim=-1, index=self.prim_ids_tensor)  # [B,K]
        prim_logits = prim_logits / max(1e-6, self.temperature)
        probs = torch.softmax(prim_logits, dim=-1)  # [B,K]

        # 组装成 [B,P,K]
        probs_full = torch.full((B, P, self.K), 1.0 / self.K, device=self.device, dtype=probs.dtype)
        for b in range(B):
            mpos = int(chosen_pos[b].item())
            if mpos >= 0:
                probs_full[b, mpos, :] = probs[b]

        return TeacherOut(probs_full, chosen_pos)

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

        self.student = StrongStudent(
            seq_len_pad=self.seq_len_pad,  # Pass Padded Length
            patch_len=self.patch_len,
            in_channels=self.C,
            dim_model=self.dim_student,
            num_vq_codes=self.num_primitives,
            nhead=4,
            num_layers=4
        ).to(self.device)
        self.mask_rate=args.mask_rate
        # --- 4. Initialize Teacher 2 (SoftLlama) ---
        self.lambda_distill = float(getattr(args, "lambda_distill", 1))
        self.teacher = None

        # if self.stage == 1 and hasattr(args, 'llama_name11'):
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
            adapter_path = r"D:\fuy\MyCode\SensorLLMLib_v2\runs\SensorLoRA\mhealth\alignment_weight\stage2\ckpts\lora_MHealth_SensorLoRA_MHealth_ftM_sl100_ll48_pl0_dm32_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_test_0"
            adapter_path = r"D:\fuy\MyCode\SensorLLMLib_v2\runs\lora_UCIHAR_SensorLoRA_UCIHAR_ftM_sl128_ll48_pl0_dm32_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_test_0"
            # adapter_path = r"D:\fuy\MyCode\SensorLLMLib_v2\runs\SensorLoRA_next_token\mhealth\alignment_weight\stage2\ckpts\lora_MHealth_SensorLoRA_MHealth_ftM_sl100_ll48_pl0_dm32_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_test_0"
            self.teacher = SoftLlamaTeacherLoRA(
                llm_path=args.llama_name,
                adapter_path=adapter_path,
                mask_token_id=self.mask_token_id,
                K=self.mask_token_id,
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

    def forward(self, x_imu, padding_mask=None, mode=None, labels=None):
        if mode is None:
            mode = "pretrain" if self.stage == 1 else "classify"

        if not torch.is_tensor(x_imu): x_imu = torch.as_tensor(x_imu)
        B, L_orig, C = x_imu.shape

        # ===========================================================
        # Step 0: 统一 Padding (关键!)
        # 1. Pad 输入数据到 VQ Stride 的倍数
        # ===========================================================
        x_pad, L_pad = _pad_to_multiple(x_imu, self.vq_stride, pad_value=0.0)
        # x_pad: [B, seq_len_pad, C]

        # 2. 生成 Loss Mask (用于忽略 Padding 区域的 Loss)
        # L_orig 是原始有效长度。计算有多少个 Patch 是有效的。
        # 例如: Orig=195, Stride=8 -> 24 个有效 Patch (24*8=192), 第 25 个 Patch 包含填充数据
        valid_patches = L_orig // self.patch_len
        loss_valid_mask = torch.zeros((B, self.P), device=self.device, dtype=torch.bool)
        loss_valid_mask[:, :valid_patches] = True

        # ====================
        # Stage 1: Pretrain
        # ====================
        if self.stage == 1:
            with torch.no_grad():
                # VQ-VAE 的 get_token_ids 必须接收 Pad 后的数据
                # 如果你的 VQ-VAE 内部没有自动 Pad，这里传 x_pad 是最安全的
                gt_ids = self.vq_net.get_token_ids(x_pad)

                # 安全断言：确保 VQ 输出的 Token 数与 Student 的 Patch 数对齐
            if gt_ids.shape[1] != self.P:
                # 容错：如果 VQ 内部处理导致稍微多了点，强行截断对齐
                if gt_ids.shape[1] > self.P:
                    gt_ids = gt_ids[:, :self.P]
                else:
                    raise ValueError(f"VQ Tokens {gt_ids.shape[1]} < Student Patches {self.P}")

            # 生成 Masking (BEiT 任务)
            mask_bool = self.random_masking(x_pad.size(0), self.P, self.mask_rate)
            # mask_bool = self.random_masking(x_pad.size(0), self.P, 0.4)

            # Student Forward
            student_logits, _ = self.student(x_pad, mask_bool)  # [B, P, K]

            # --- Calculation with Loss Mask ---
            # 只有 (被 Mask 的位置) AND (不是 Padding 的位置) 才计算 Loss
            final_mask = mask_bool & loss_valid_mask

            target_masked = gt_ids[final_mask]
            pred_masked = student_logits[final_mask]

            if target_masked.numel() > 0:
                loss_recon = F.cross_entropy(pred_masked, target_masked)
            else:
                loss_recon = torch.tensor(0.0, device=self.device, requires_grad=True)

            loss_distill = torch.tensor(0.0, device=self.device)

            # 用于监控的统计指标初始化
            teacher_acc_batch = 0.0
            teacher_keep_ratio = 0.0

            if self.teacher is not None and self.lambda_distill > 0:
                # 1. 构造 Teacher 输入 (把 mask 位置换成 mask_token_id)
                teacher_input_ids = gt_ids.clone()
                teacher_input_ids[mask_bool] = self.mask_token_id

                # 2. Teacher 推理
                # 注意：这里 teacher_out.probs 已经是 [B, P, K] 的稀疏矩阵了
                teacher_out = self.teacher(teacher_input_ids, pick_from_mask=final_mask)

                # 3. 构造 distill_mask (只针对 Teacher 挑选出的那个位置)
                # chosen_pos 是 [B], 存的是每个样本被选中的那个 mask 的 index
                chosen_mask = torch.zeros_like(final_mask, dtype=torch.bool)
                for b in range(chosen_mask.size(0)):
                    p = int(teacher_out.chosen_pos[b].item())
                    if p >= 0:
                        chosen_mask[b, p] = True

                # 最终用于计算蒸馏 loss 的 mask (必须是 Teacher 选中的 AND Student Mask 的 AND 非 Padding 的)
                distill_mask = final_mask & chosen_mask

                if distill_mask.any():
                    # 4. 获取 Teacher 和 Student 的 Logits/Probs
                    s_logp = F.log_softmax(student_logits[distill_mask], dim=-1)  # [M, K]
                    t_probs = teacher_out.probs[distill_mask].detach()  # [M, K]
                    gt_sel = gt_ids[distill_mask]  # [M] (Ground Truth)

                    # =====================================================
                    # 【核心修正】：双重门控 (Double Gating)
                    # =====================================================

                    # (A) 自信度门控: Teacher 必须足够自信
                    conf_th = float(getattr(self.args, "teacher_conf_th", 0.02))
                    is_confident = t_probs.max(dim=-1).values > conf_th

                    # (B) 正确性门控: Teacher 的 Top-1 必须是对的！(或者 Top-3 包含 GT)
                    # 对于 1B 这种小模型，建议严格一点，只信任它预测正确的时候
                    teacher_pred = t_probs.argmax(dim=-1)
                    is_correct = (teacher_pred == gt_sel)

                    # 只有既自信又正确的样本，才用来蒸馏
                    keep = is_confident & is_correct

                    # --- 统计指标 ---
                    teacher_acc_batch = float(is_correct.float().mean().item())  # 监控 Teacher 有多准
                    teacher_keep_ratio = float(keep.float().mean().item())  # 监控有多少样本参与了蒸馏

                    if keep.any():
                        # (C) 计算 KL Loss
                        # 只有 keep 为 True 的行才计算
                        # scale_factor: 平衡 loss 数量级。因为 recon 是 P 个点，distill 只有不到 1 个点
                        # 简单起见，可以先不加 scale，或者手动把 lambda_distill 调大 (比如 5.0)
                        loss_distill = F.kl_div(
                            s_logp[keep],
                            t_probs[keep],
                            reduction="batchmean"
                        )

            loss_total = loss_recon + self.lambda_distill * loss_distill

            metrics = {
                "loss_recon": float(loss_recon.item()),
                "loss_distill": float(loss_distill.item()),
                "loss_total": float(loss_total.item()),
                "teacher_acc": teacher_acc_batch,  # <--- 重点看这个！
                "teacher_keep_ratio": teacher_keep_ratio
            }
            # print(metrics)
            return loss_total, student_logits, metrics
        # ====================
        # Stage 2: Classify
        # ====================
        elif self.stage == 2:
            # Student Forward
            _, feat = self.student(x_pad, mask_bool=None)  # [B, P, D]

            # --- Pooling with Mask ---
            # 只对有效的 Patch 进行平均，忽略 Padding 部分
            # feat: [B, P, D] -> [B, valid_patches, D]
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