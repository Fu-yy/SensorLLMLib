import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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


class SoftLlamaTeacher(nn.Module):
    """
    Ablation switches:
      - teacher_init: "pretrained" | "random"
      - use_prompt: bool
      - load_adapter: bool  (projector + output_head)
    """
    def __init__(
        self,
        llm_path,
        adapter_path,
        codebook_weights,
        mask_token_id,
        device,
        teacher_init="pretrained",
        use_prompt=True,
        load_adapter=True,
        system_prompt="Analyze sensor sequence:"
    ):
        super().__init__()
        self.device = device
        self.mask_token_id = mask_token_id
        self.use_prompt = bool(use_prompt)

        # ---- LLM init mode ----
        if teacher_init == "pretrained":
            print(f"[Teacher] Loading PRETRAINED Llama from {llm_path}...")
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path, torch_dtype=torch.float16, trust_remote_code=True
            ).to(device).eval()
        elif teacher_init == "random":
            print(f"[Teacher] Initializing RANDOM Llama (same config) from {llm_path}...")
            cfg = AutoConfig.from_pretrained(llm_path, trust_remote_code=True)
            self.llm = AutoModelForCausalLM.from_config(cfg).to(device).eval()
            # random init -> keep fp16? 你也可以转 fp16，但数值稳定性可能更差
            self.llm = self.llm.to(dtype=torch.float16)
        else:
            raise ValueError(f"Unknown teacher_init={teacher_init}")

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        for p in self.llm.parameters():
            p.requires_grad = False

        self.llm_dim = self.llm.config.hidden_size

        # codebook: [K, D_vq]
        self.register_buffer("codebook", codebook_weights.to(device))
        self.num_vq_codes = self.codebook.shape[0]
        self.vq_dim = self.codebook.shape[1]

        # projector + head
        self.projector = nn.Linear(self.vq_dim, self.llm_dim).to(device)
        self.mask_embed_llama = nn.Parameter(torch.randn(1, 1, self.llm_dim).to(device))
        self.output_head = nn.Linear(self.llm_dim, self.num_vq_codes).to(device)

        # ---- prompt embeds (optional) ----
        self.system_prompt = system_prompt
        if self.use_prompt and self.system_prompt:
            prompt_input_ids = self.tokenizer(self.system_prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                self.prompt_embeds = self.llm.get_input_embeddings()(prompt_input_ids)
        else:
            self.prompt_embeds = None

        # ---- adapter loading (optional) ----
        # adapter 仅指 projector/output_head 的对齐权重
        if load_adapter and adapter_path is not None and os.path.exists(adapter_path):
            print(f"[Teacher] Loading Adapter (projector/head) from {adapter_path}")
            ckpt = torch.load(adapter_path, map_location=device)
            # 你自己的 ckpt key 可能不同：请确保这里对应上
            if "projector" in ckpt:
                self.projector.load_state_dict(ckpt["projector"], strict=True)
            if "output_head" in ckpt:
                self.output_head.load_state_dict(ckpt["output_head"], strict=True)

            for p in self.projector.parameters(): p.requires_grad = False
            for p in self.output_head.parameters(): p.requires_grad = False
        else:
            if load_adapter:
                print("[Teacher] WARNING: Adapter not loaded (projector/head random).")
            # 即便不 load，也要冻结（teacher 训练阶段不更新）
            for p in self.projector.parameters(): p.requires_grad = False
            for p in self.output_head.parameters(): p.requires_grad = False

    @torch.no_grad()
    def forward(self, masked_ids):
        """
        masked_ids: [B, T], where masked positions are == mask_token_id
        return: probs [B, T, K]
        """
        B, T = masked_ids.shape
        is_mask = (masked_ids == self.mask_token_id)

        # safe ids for embedding lookup
        safe_ids = masked_ids.clone()
        safe_ids[is_mask] = 0
        safe_ids = safe_ids.clamp(0, self.num_vq_codes - 1)

        # VQ embeddings -> projector -> llama space
        vq_embeds = F.embedding(safe_ids, self.codebook)              # [B, T, D_vq]
        sensor_embeds = self.projector(vq_embeds)                     # [B, T, D_llm]

        # replace masked positions with learnable mask embedding
        expanded_mask = self.mask_embed_llama.expand(B, T, -1)         # [B, T, D_llm]
        m = is_mask.unsqueeze(-1).float()
        final_sensor_embeds = sensor_embeds * (1 - m) + expanded_mask * m

        # optional prompt
        if self.prompt_embeds is not None:
            batch_prompt = self.prompt_embeds.expand(B, -1, -1)        # [B, Lp, D_llm]
            inputs_embeds = torch.cat([batch_prompt, final_sensor_embeds], dim=1).to(self.llm.dtype)
            L_text = batch_prompt.shape[1]
        else:
            inputs_embeds = final_sensor_embeds.to(self.llm.dtype)
            L_text = 0

        out = self.llm(inputs_embeds=inputs_embeds, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]                            # [B, L_text+T, D_llm]
        sensor_features = last_hidden[:, L_text:, :]                   # [B, T, D_llm]

        logits = self.output_head(sensor_features.to(torch.float32))   # [B, T, K]
        probs = F.softmax(logits, dim=-1)

        return probs
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# ------------------------------
# 1) Backbone: Vanilla Transformer
# ------------------------------
class VanillaTransformerBackbone(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, ffn_mult=4, dropout=0.0, norm_first=True):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x):  # x: [B, T, D]
        return self.encoder(x)  # [B, T, D]


# ------------------------------
# 2) Backbone: Linear / MLP (no attention)
# ------------------------------
class LinearBackbone(nn.Module):
    def __init__(self, d_model=512, mlp_layers=1, hidden_dim=1024, dropout=0.0):
        super().__init__()
        if mlp_layers == 1:
            self.net = nn.Identity()
        elif mlp_layers == 2:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
            )
        else:
            raise ValueError("mlp_layers should be 1 or 2")

    def forward(self, x):
        return self.net(x)


# ------------------------------
# 3) Unified Teacher (select backbone by one arg)
# ------------------------------
class UnifiedTeacher(nn.Module):
    """
    teacher_backbone: "llama" | "transformer" | "linear"

    Input : masked_ids [B, T] where masked positions == mask_token_id
    Output: probs [B, T, K]
    """
    def __init__(
        self,
        teacher_backbone: str,
        codebook_weights: torch.Tensor,   # [K, D_vq]
        mask_token_id: int,
        device: torch.device,

        # shared dims
        vq_dim: int = None,

        # llama args
        llm_path: str = None,
        teacher_init: str = "pretrained",     # "pretrained" | "random" (for llama)
        use_prompt: bool = True,
        system_prompt: str = "Analyze sensor sequence:",

        # transformer teacher args
        teacher_d_model: int = 512,
        teacher_nhead: int = 8,
        teacher_num_layers: int = 6,
        teacher_ffn_mult: int = 4,
        teacher_dropout: float = 0.0,
        teacher_norm_first: bool = True,

        # linear teacher args
        teacher_mlp_layers: int = 1,
        teacher_hidden_dim: int = 1024,

        # freezing behavior
        teacher_trainable: bool = False,
    ):
        super().__init__()
        self.teacher_backbone = str(teacher_backbone).lower()
        self.device = device
        self.mask_token_id = int(mask_token_id)

        # codebook
        self.register_buffer("codebook", codebook_weights.to(device))
        self.num_vq_codes = self.codebook.shape[0]
        self.vq_dim = self.codebook.shape[1] if vq_dim is None else int(vq_dim)

        # ==== Choose teacher hidden size (D_teacher) ====
        # - llama: D_teacher = llm_hidden
        # - transformer/linear: D_teacher = teacher_d_model
        self.use_prompt = bool(use_prompt)
        self.system_prompt = system_prompt

        self.llm = None
        self.tokenizer = None
        self.prompt_embeds = None

        if self.teacher_backbone == "llama":
            if llm_path is None:
                raise ValueError("llm_path must be provided when teacher_backbone='llama'")

            if teacher_init == "pretrained":
                print(f"[Teacher] Loading PRETRAINED Llama from {llm_path}...")
                self.llm = AutoModelForCausalLM.from_pretrained(
                    llm_path, torch_dtype=torch.float16, trust_remote_code=True
                ).to(device).eval()
            elif teacher_init == "random":
                print(f"[Teacher] Initializing RANDOM Llama (same config) from {llm_path}...")
                cfg = AutoConfig.from_pretrained(llm_path, trust_remote_code=True)
                self.llm = AutoModelForCausalLM.from_config(cfg).to(device).eval()
                self.llm = self.llm.to(dtype=torch.float16)
            else:
                raise ValueError(f"Unknown teacher_init={teacher_init}")

            for p in self.llm.parameters():
                p.requires_grad = False

            self.teacher_dim = self.llm.config.hidden_size  # D_teacher

            # optional prompt
            self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
            if self.use_prompt and self.system_prompt:
                prompt_ids = self.tokenizer(self.system_prompt, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    self.prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
            else:
                self.prompt_embeds = None

            self.backbone = None  # llama backbone is self.llm

        elif self.teacher_backbone == "transformer":
            self.teacher_dim = int(teacher_d_model)
            self.backbone = VanillaTransformerBackbone(
                d_model=self.teacher_dim,
                nhead=int(teacher_nhead),
                num_layers=int(teacher_num_layers),
                ffn_mult=int(teacher_ffn_mult),
                dropout=float(teacher_dropout),
                norm_first=bool(teacher_norm_first),
            ).to(device)

            # transformer teacher 不需要 prompt（你也可以保留，但会多引入一个“文本 token”概念）
            self.prompt_embeds = None

        elif self.teacher_backbone == "linear":
            self.teacher_dim = int(teacher_d_model)
            self.backbone = LinearBackbone(
                d_model=self.teacher_dim,
                mlp_layers=int(teacher_mlp_layers),
                hidden_dim=int(teacher_hidden_dim),
                dropout=float(teacher_dropout),
            ).to(device)
            self.prompt_embeds = None

        else:
            raise ValueError(f"Unknown teacher_backbone={self.teacher_backbone}")

        # ==== projector: VQ space -> teacher hidden space ====
        self.projector = nn.Linear(self.vq_dim, self.teacher_dim).to(device)

        # ==== mask embedding in teacher hidden space ====
        self.mask_embed = nn.Parameter(torch.randn(1, 1, self.teacher_dim, device=device))

        # ==== output head: teacher hidden -> VQ vocab logits ====
        self.output_head = nn.Linear(self.teacher_dim, self.num_vq_codes).to(device)

        # ---- freeze teacher if desired ----
        if not teacher_trainable:
            for p in self.projector.parameters():
                p.requires_grad = False
            for p in self.output_head.parameters():
                p.requires_grad = False
            if self.backbone is not None:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            # llama 已经冻结过

    @torch.no_grad()
    def forward(self, masked_ids: torch.Tensor):
        """
        masked_ids: [B, T] with mask_token_id
        return probs: [B, T, K]
        """
        B, T = masked_ids.shape
        is_mask = (masked_ids == self.mask_token_id)

        safe_ids = masked_ids.clone()
        safe_ids[is_mask] = 0
        safe_ids = safe_ids.clamp(0, self.num_vq_codes - 1)

        # VQ ids -> embeddings -> teacher hidden
        vq_embeds = F.embedding(safe_ids, self.codebook)          # [B, T, D_vq]
        h = self.projector(vq_embeds)                             # [B, T, D_teacher]

        # replace masked positions with learned mask embedding
        m = is_mask.unsqueeze(-1).float()
        h = h * (1 - m) + self.mask_embed.expand(B, T, -1) * m

        # backbone
        if self.teacher_backbone == "llama":
            if self.prompt_embeds is not None:
                prompt = self.prompt_embeds.expand(B, -1, -1)      # [B, Lp, D]
                inputs_embeds = torch.cat([prompt, h], dim=1).to(self.llm.dtype)
                Lp = prompt.shape[1]
            else:
                inputs_embeds = h.to(self.llm.dtype)
                Lp = 0

            out = self.llm(inputs_embeds=inputs_embeds, output_hidden_states=True)
            last_hidden = out.hidden_states[-1]                    # [B, Lp+T, D]
            h2 = last_hidden[:, Lp:, :].to(torch.float32)          # [B, T, D]

        else:
            h2 = self.backbone(h)                                  # [B, T, D]
            h2 = h2.to(torch.float32)

        logits = self.output_head(h2)                               # [B, T, K]
        probs = F.softmax(logits, dim=-1)
        return probs


# ==============================================================================
# Model: modify stage-1 loss for ablations
# ==============================================================================
def distill_loss(student_logits, teacher_probs, mask, distill_type="soft_kl", temperature=1.0):
    """
    student_logits: [N, K] (masked positions flattened)
    teacher_probs:  [N, K] (masked positions flattened)
    mask already applied outside.
    """
    if distill_type == "none":
        return torch.tensor(0.0, device=student_logits.device)

    if temperature is None:
        temperature = 1.0
    T = float(temperature)

    if distill_type == "soft_kl":
        # KL(teacher || student)
        s_logp = F.log_softmax(student_logits / T, dim=-1)
        t_p = (teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)).clamp_min(1e-8)
        # batchmean: stable
        return F.kl_div(s_logp, t_p, reduction="batchmean") * (T * T)

    if distill_type == "hard_ce":
        # teacher top1 as hard label
        hard = teacher_probs.argmax(dim=-1)  # [N]
        return F.cross_entropy(student_logits, hard)

    raise ValueError(f"Unknown distill_type={distill_type}")


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

        self.use_teacher = bool(getattr(args, "use_teacher", True))
        self.teacher_init = str(getattr(args, "teacher_init", "pretrained"))  # pretrained | random
        self.teacher_use_prompt = bool(getattr(args, "teacher_use_prompt", True))
        self.teacher_load_adapter = bool(getattr(args, "teacher_load_adapter", True))

        self.distill_type = str(getattr(args, "distill_type", "soft_kl"))  # soft_kl | hard_ce | none
        self.distill_temperature = float(getattr(args, "distill_temperature", 1.0))

        if self.stage == 1 and getattr(args, "llama_name", None) and self.use_teacher:
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
            teacher_backbone = str(getattr(args, "teacher_backbone", "transformer")).lower()  # "llama"|"transformer"|"linear"

            self.teacher = UnifiedTeacher(
                teacher_backbone=teacher_backbone,
                codebook_weights=codebook_weights,
                mask_token_id=self.mask_token_id,
                device=self.device,

                # llama
                llm_path=getattr(args, "llama_name", None),
                teacher_init=str(getattr(args, "teacher_init", "pretrained")).lower(),
                use_prompt=bool(getattr(args, "teacher_use_prompt", True)),
                system_prompt=str(getattr(args, "system_prompt", "Analyze sensor sequence:")),

                # transformer/linear
                teacher_d_model=int(getattr(args, "teacher_d_model", 512)),
                teacher_nhead=int(getattr(args, "teacher_nhead", 8)),
                teacher_num_layers=int(getattr(args, "teacher_num_layers", 6)),
                teacher_ffn_mult=int(getattr(args, "teacher_ffn_mult", 4)),
                teacher_dropout=float(getattr(args, "teacher_dropout", 0.0)),
                teacher_norm_first=bool(getattr(args, "teacher_norm_first", True)),

                teacher_mlp_layers=int(getattr(args, "teacher_mlp_layers", 1)),
                teacher_hidden_dim=int(getattr(args, "teacher_hidden_dim", 1024)),

                teacher_trainable=bool(getattr(args, "teacher_trainable", False)),
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
        V = max(0, valid_patches)

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
            student_logits, _ = self.student(x_pad, None)  # [B, P, K]

            # --- Calculation with Loss Mask ---
            # 只有 (被 Mask 的位置) AND (不是 Padding 的位置) 才计算 Loss
            # final_mask = mask_bool & loss_valid_mask

            target_masked = gt_ids[:, 1:V]
            pred_masked = student_logits[:, :V - 1, :]  # [B,V-1,K]

            if target_masked.numel() > 0:
                loss_recon = F.cross_entropy(pred_masked.reshape(-1, pred_masked.size(-1)), target_masked.reshape(-1))
            else:
                loss_recon = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Distill Loss
            loss_distill = torch.tensor(0.0, device=self.device)

            if self.teacher is not None and self.lambda_distill > 0 and self.distill_type != "none":
                teacher_input_ids = gt_ids.clone()
                # teacher_input_ids[mask_bool] = self.mask_token_id

                with torch.no_grad():
                    t_probs_full = self.teacher(teacher_input_ids)  # [B, P, K]

                # apply final_mask
                s_flat = student_logits  # [N, K]
                t_flat = t_probs_full.detach()  # [N, K]

                if s_flat.numel() > 0:
                    loss_distill = distill_loss(
                        s_flat, t_flat, mask=None,
                        distill_type=self.distill_type,
                        temperature=self.distill_temperature
                    )

            loss_total = loss_recon + self.lambda_distill * loss_distill

            return loss_total, student_logits, {
                "loss_recon": loss_recon.item(),
                "loss_distill": loss_distill.item(),
                "loss_total": loss_total.item()
            }

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