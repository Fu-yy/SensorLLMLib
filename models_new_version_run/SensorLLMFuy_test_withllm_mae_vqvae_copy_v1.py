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


class SoftLlamaTeacher(nn.Module):
    """
    Soft-Prompting based LLM Teacher for Time-Series.
    Aligns VQ-VAE tokens into LLM embedding space via a linear projector.
    """

    def __init__(self, llm_path, codebook_weights, mask_token_id, device):
        super().__init__()
        self.device = device
        self.mask_token_id = mask_token_id  # 比如 512

        # ====================================================
        # 1. Load Frozen Llama (The "Brain")
        # ====================================================
        print(f"[Teacher] Loading Llama from {llm_path}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        # 彻底冻结 Llama 本体
        for p in self.llm.parameters():
            p.requires_grad = False

        self.llm_dim = self.llm.config.hidden_size

        # ====================================================
        # 2. Load VQ Codebook (The "Dictionary")
        # ====================================================
        print(f"[Teacher] Loading Codebook from {codebook_weights}...")
        # 假设 ckpt 是直接保存的 tensor 或者 state_dict
        # 如果是 state_dict，请根据实际 key 修改，这里假设加载出来就是 [K, D_vq]

        # 注册为 buffer (不更新 codebook，只查表)
        self.register_buffer("codebook", codebook_weights.to(device))
        self.num_vq_codes = self.codebook.shape[0]
        self.vq_dim = self.codebook.shape[1]

        # ====================================================
        # 3. Learnable Components (The "Adapter") - 论文核心
        # ====================================================
        # A. Projector: VQ Space -> LLM Space
        self.projector = nn.Linear(self.vq_dim, self.llm_dim).to(device)

        # B. Learnable Mask Token: Llama 空间中的特殊 Mask 向量
        # 形状为 [1, 1, llm_dim]，用于替换被 mask 的位置
        self.mask_embed_llama = nn.Parameter(torch.randn(1, 1, self.llm_dim).to(device))

        # C. Output Head: LLM Space -> VQ Probability
        # 用于将 Llama 的理解映射回 Sensor Token 的概率，以便计算蒸馏 Loss
        self.output_head = nn.Linear(self.llm_dim, self.num_vq_codes).to(device)

        # ====================================================
        # 4. System Prompt (Context Priming)
        # ====================================================
        self.system_prompt = "Analyze the following sensor activity sequence and predict the masked values:\n"
        self.prompt_input_ids = self.tokenizer(self.system_prompt, return_tensors="pt").input_ids.to(device)
        # 预计算 Prompt Embedding 以节省时间
        with torch.no_grad():
            self.prompt_embeds = self.llm.get_input_embeddings()(self.prompt_input_ids)  # [1, L_text, D_llm]

    def forward(self, masked_ids):
        """
        Args:
            masked_ids: [B, T] Integer tensor containing:
                        - 0~511: Valid VQ tokens
                        - 512 (self.mask_token_id): Mask positions
        """
        B, T = masked_ids.shape

        # -------------------------------------------------------
        # Step 1: Prepare Sensor Embeddings (Soft Prompting)
        # -------------------------------------------------------
        # 1.1 识别 Mask 位置
        is_mask = (masked_ids == self.mask_token_id)  # [B, T] Bool

        # 1.2 处理 ID 越界问题：先把 Mask ID 替换成 0 (为了能查表)，稍后覆盖
        safe_ids = masked_ids.clone()
        safe_ids[is_mask] = 0
        safe_ids = safe_ids.clamp(0, self.num_vq_codes - 1)

        # 1.3 查表获取 VQ 向量 [B, T, vq_dim]
        vq_embeds = F.embedding(safe_ids, self.codebook)

        # 1.4 投影到 LLM 空间 [B, T, llm_dim]
        sensor_embeds = self.projector(vq_embeds)

        # 1.5 【关键】用可学习的 Mask 向量覆盖 Mask 位置
        # 扩展 mask_embed 到 [B, T, llm_dim]
        expanded_mask = self.mask_embed_llama.expand(B, T, -1)
        # 使用 where 进行替换：如果是 mask，用 mask_embed，否则用 projected_vq
        # unsqueeze mask to [B, T, 1] for broadcasting
        mask_broadcast = is_mask.unsqueeze(-1).float()
        final_sensor_embeds = sensor_embeds * (1 - mask_broadcast) + expanded_mask * mask_broadcast

        # -------------------------------------------------------
        # Step 2: Combine with Text Prompt
        # -------------------------------------------------------
        # prompt_embeds: [1, L_text, D] -> [B, L_text, D]
        batch_prompt_embeds = self.prompt_embeds.expand(B, -1, -1)

        # Concatenate: [B, L_text + T, D]
        inputs_embeds = torch.cat([batch_prompt_embeds, final_sensor_embeds], dim=1)
        inputs_embeds = inputs_embeds.to(self.llm.dtype)
        # -------------------------------------------------------
        # Step 3: Llama Inference
        # -------------------------------------------------------
        # 注意：这里我们让 Llama 处理整个序列。
        # 如果是训练 Projector 阶段，这里要有梯度。
        # 如果 Projector 已经训好，或者是纯蒸馏，可以用 no_grad。
        # 假设我们在训练 Adapter (Projector + Output Head):
        # 这样模型才会返回中间层结果，不仅仅是 logits
        outputs = self.llm(inputs_embeds=inputs_embeds, output_hidden_states=True)

        # Llama Outputs: outputs.hidden_states 是一个元组，包含每一层的输出
        # hidden_states[-1] 代表最后一层输出 (Before LM Head)，形状为 [B, L_total, llm_dim]
        last_hidden_state = outputs.hidden_states[-1]

        # 我们只关心对应 Sensor 部分的输出，去掉 Text Prompt 部分
        L_text = batch_prompt_embeds.shape[1]
        sensor_features = last_hidden_state[:, L_text:, :]  # [B, T, llm_dim]

        # -------------------------------------------------------
        # Step 4: Map back to VQ Probability (For Distillation)
        # -------------------------------------------------------
        # 关键修复3：确保转回 float32 计算 logits，防止溢出
        logits = self.output_head(sensor_features.to(torch.float32))

        # Softmax 得到概率分布
        probs = F.softmax(logits, dim=-1)

        valid = torch.ones(B, dtype=torch.bool, device=self.device)

        return TeacherOut(probs=probs, valid=valid)
# ==========================================
# 3. Student Network (你需要训练的小模型)
# ==========================================

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


# -------------------------
# 2) Model: fix P, fix num_latents, add no_resampler option
# -------------------------
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

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
        # ===== You already have ds_cfg logic; keep it =====
        # self.C = int(getattr(args, "enc_in", 15))
        # self.num_class = int(getattr(args, "num_class", 12))
        self.stage = int(getattr(args, "stage", 1))

        self.seq_len = int(getattr(args, "seq_len", 200))
        self.patch_len = int(getattr(args, "patch_len", 20))  # IMPORTANT: make P >= 8
        assert self.seq_len % self.patch_len == 0
        self.P = self.seq_len // self.patch_len
        self.mask_rate = float(getattr(args, "mask_rate", 0.5))  # MAE usually 0.4~0.75; with HAR try 0.4~0.6

        self.resampler_dim = int(getattr(args, "resampler_dim", 256))  # smaller is fine
        # IMPORTANT: latents should NOT exceed P massively
        self.num_latents = int(getattr(args, "num_latents", min(self.P, 16)))


        # --------------------------------------------

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

        # 获取一些关键参数
        self.num_primitives = getattr(self.vq_net, 'code_num', 512)  # K
        self.mask_token_id = self.num_primitives  # Mask ID 设为 K (0~K-1是有效token)

        # --------------------------------------------
        # B. 初始化 Student (小模型) - 训练
        # --------------------------------------------
        codebook_weights = torch.load(self.qua_path + os.sep + "best_codebook.pth", map_location="cpu")
        # 根据你的保存格式，可能是 ckpt['embedding'] 或直接是 ckpt
        # codebook_weights = ckpt['model']['quantizer.embedding.weight']  # 举例，需根据实际情况调整
        num_codes, code_dim = codebook_weights.shape
        dim_student = int(getattr(args, "dim_student", 256))
        dim_student = code_dim
        self.student = StudentTransformer(
            codebook_weights=codebook_weights,
            num_tokens=self.num_primitives,
            dim_model=dim_student,
            nhead=4,
            num_layers=4,
            max_len=getattr(self.vq_net, 'seq_len', 200)  # 假设 VQ 输出长度
        ).to(self.device)

        # --------------------------------------------
        # C. 初始化 Teacher (LLM) - 冻结
        # --------------------------------------------
        # ----------------------------------------------------
        # Stage 1 独有: Teacher & Masking Params
        # ----------------------------------------------------
        if self.stage == 1:
            self.use_llm_teacher = getattr(args, "use_llm_teacher", "soft")
            self.teacher = None

            if self.use_llm_teacher == "soft":
                llama_name = getattr(args, "llama_name", "meta-llama/Llama-2-7b-chat-hf")
                self.teacher = SoftLlamaTeacher(
                    llm_path=llama_name,
                    codebook_weights=codebook_weights,
                    mask_token_id=self.mask_token_id,
                    device=self.device
                )
            else:
                llama_name = getattr(args, "llama_name", "meta-llama/Llama-2-7b-chat-hf")
                self.teacher = PrimitiveLLMTeacher(
                    llm_name_or_path=llama_name,
                    K=self.num_primitives,
                    device=self.device
                )
        # ----------------------------------------------------
        # Stage 2 独有: Classifier Head
        # ----------------------------------------------------
        elif self.stage == 2:
            self.teacher = None  # 关键：不加载 LLM，省显存
            self.classifier = nn.Sequential(
                nn.Linear(dim_student, dim_student),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim_student, self.num_class)
            )
        # --------------------------------------------
        # D. Loss 权重
        # --------------------------------------------
        self.lambda_recon = 1.0
        self.lambda_distill = float(getattr(args, "lambda_distill", 0.5))


    def random_masking(self, x, mask_ratio):
        """
        BERT-style random masking.
        x: [B, T] token ids
        return: masked_x, mask_bool (True where masked)
        """
        B, T = x.shape
        noise = torch.rand(B, T, device=x.device)

        # 创建 mask (mask_ratio 的概率被 mask)
        mask_bool = noise < mask_ratio

        # 创建 masked input
        masked_x = x.clone()
        masked_x[mask_bool] = self.mask_token_id  # 替换为 mask token

        return masked_x, mask_bool


    def forward(self, x_imu, padding_mask=None, mode: Optional[str] = None, labels=None):
        if mode is None:
            mode = "pretrain" if self.stage == 1 else "classify"

        if not torch.is_tensor(x_imu):
            x_imu = torch.as_tensor(x_imu)

        # 1. VQ-VAE Tokenization (获得 Ground Truth Tokens)
        with torch.no_grad():
            # 假设 vq_net 有 get_token_ids 方法，返回 [B, T_code]
            gt_ids = self.vq_net.get_token_ids(x_imu)
            # 确保 gt_ids 在 0 ~ K-1 之间
        # ==========================================
        # Branch 1: Stage 1 (Pre-training)
        # ==========================================
        if self.stage == 1:
        # 2. Random Masking
            # mask_rate 可以从 args 读取
            mask_rate = getattr(self.args, 'mask_rate', 0.4)
            masked_ids, mask_bool = self.random_masking(gt_ids, mask_rate)

            # 3. Student Forward (预测)
            # student_logits: [B, T, K]
            student_logits, student_feat = self.student(masked_ids)

            # -------------------------
            # 4. 计算 Loss
            # -------------------------
            loss_dict = {}

            # A. Reconstruction Loss (CE): Student 预测结果 vs 真实 VQ Token
            # 只计算被 mask 的部分
            # flatten for CE loss
            target_masked = gt_ids[mask_bool]  # [N_masked]
            pred_masked = student_logits[mask_bool]  # [N_masked, K]

            loss_recon = F.cross_entropy(pred_masked, target_masked)
            loss_dict['recon_loss'] = loss_recon

            loss_distill = torch.tensor(0.0, device=self.device)

            # B. Distillation Loss (KL): Student 分布 vs Teacher 分布
            # 只有在 Teacher 启用时计算
            if self.teacher is not None and self.lambda_distill > 0:
                # 这里的策略是：LLM 太慢，如果对每个 Batch 都算太耗时。
                # 策略：随机选取 Batch 中的一部分或者只对 Mask 的中心位置进行 Teacher 指导。
                # 为了代码简单，这里演示对整个 batch 的 masked_ids 进行指导。

                # 注意：LLM 输入的是 masked_ids (包含 <PMASK>)
                # TeacherOut.probs 是 [B, K] (LLM 预测 Prompt 中 <PMASK> 处的词)
                # 但我们的 masked_ids 可能有多个 mask。
                # 这是一个关键点：标准的 LLM 补全是 Seq2Seq 或 Causal。
                # 简单起见，我们假设 Teacher 只预测序列中"整体语义"或者我们构造特定的 Prompt
                # 这里的实现：为了让 Teacher 运行，我们可能需要简化。
                # 比如：只把含有 Mask 的序列喂进去，并且让 Teacher 预测被 Mask 的那些 Token 的混合分布。

                # 【优化版交互】：
                # 为了让 LLM 高效，我们只拿 Batch 中第一个 Mask 位置询问 Teacher
                # 或者，如果显存允许，通过 prompt 让 Teacher 预测整个序列 (seq2seq)。
                # 下面是使用之前定义的 PrimitiveLLMTeacher (Next Token Prediction 风格) 的逻辑：

                teacher_out = self.teacher(masked_ids)  # [B, K]

                # 我们假设 Teacher 返回的是对于当前 Masked Sequence 最可能的 "Next Token" 或者 "Filling"
                # 我们把 Student 在 mask 位置的平均 logits 或者 max logits 与 Teacher 对齐

                # 取 Student 在所有 mask 位置的平均分布 (Pooling) 来和 Teacher 对齐
                # 或者更精细：Teacher 如果只看 Prompt，它预测的是 <PMASK> 应该填什么。
                # 我们假设 masked_ids 里只有一个主要事件被 mask，或者我们取 mask_bool 的平均。

                # 简化实现：对齐 Student 在 Mask 位置的平均预测 与 Teacher 的预测
                if mask_bool.sum() > 0:
                    # 1. 提取 Student 在 Mask 位置的 Logits -> LogSoftmax
                    # 形状变成 [N_masked, K]
                    student_pred_masked = student_logits[mask_bool]
                    student_log_probs = F.log_softmax(student_pred_masked, dim=-1)  # KLDiv 要求 input 是 log_softmax

                    # 2. 提取 Teacher 在 Mask 位置的 Probs
                    # 形状变成 [N_masked, K]
                    teacher_probs_full = teacher_out.probs
                    teacher_probs_masked = teacher_probs_full[mask_bool]

                    # 确保 Teacher 不传递梯度 (虽然前面已经 detach/no_grad 了，这里双重保险)
                    teacher_probs_masked = teacher_probs_masked.detach()

                    # 3. 计算 KL 散度
                    # reduction='batchmean' 会按 batch 维度平均，这里因为已经 flatten 成了 [N_masked, K]
                    # 所以使用 default reduction (mean) 或者 'batchmean' 都可以，建议用 batchmean 配合 log_softmax
                    loss_distill = F.kl_div(student_log_probs, teacher_probs_masked, reduction='batchmean')
            loss_dict['distill_loss'] = loss_distill

            # Total Loss
            loss_total = self.lambda_recon * loss_recon + self.lambda_distill * loss_distill
            loss_dict['total_loss'] = loss_total
            log_info = {
                "P": float(self.P),
                "loss_recon": float(loss_recon.item()),
                "loss_distill": float(loss_distill.item()) if torch.is_tensor(loss_distill) else 0.0,
                "loss_total": float(loss_total.item())
            }
            return loss_total,  student_logits,log_info
        # ==========================================
        # Branch 2: Stage 2 (Fine-tuning)
        # ==========================================
        elif self.stage == 2:

            # 此时不需要 Masking，Student 能够看到完整信息
            _, features = self.student(gt_ids)  # [B, T, D]

            # Global Average Pooling
            global_feat = features.mean(dim=1)  # [B, D]

            # Classification
            logits = self.classifier(global_feat)  # [B, num_class]

            return logits

    def save_wrapper(self, path: str):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # 1. 获取当前所有参数
        state_dict = self.state_dict()

        # 2. 智能过滤 (节省空间的关键步骤)
        to_save = {}
        for key, value in state_dict.items():
            # A. 绝对不保存冻结的大模型权重 (Teacher LLM)
            # 因为它是从 HuggingFace 加载的，而且是冻结的，存下来只会浪费几GB空间
            if "teacher.llm" in key:
                continue

            # B. 绝对不保存 VQ-VAE 的权重 (如果它已经有了独立的文件 best_wrapper.pth)
            # 这样可以避免 checkpoint 变得臃肿
            # 当然，如果你希望 Student 和 VQ 绑定在一起，可以注释掉下面这两行
            if "vq_net." in key:
                continue

            to_save[key] = value

        # 3. 保存精简后的权重
        print(f"Saving model to {path} (Filtered keys: {len(state_dict) - len(to_save)})")
        torch.save(to_save, path)

    def load_wrapper(self, path: str, map_location="cpu"):
        print(f"Loading checkpoint from {path} ...")
        state_dict = torch.load(path, map_location=map_location)

        # 处理可能存在的嵌套 (比如 lightning 存的时候会有 'state_dict' 键)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        model_dict = self.state_dict()
        new_state_dict = {}

        for k, v in state_dict.items():
            # 1. 如果当前模型没有 Teacher (Stage 2)，但权重里有 Teacher，直接丢弃
            if self.stage == 2 and "teacher." in k:
                continue

            # 2. 如果当前模型不需要加载 VQ-VAE (因为它在 init 里加载过了)，丢弃
            if "vq_net." in k:
                continue

            # 3. 只有当 Key 存在于当前模型，且形状匹配时，才加载
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    new_state_dict[k] = v
                else:
                    print(f"Skipping {k}: Shape mismatch {v.shape} vs {model_dict[k].shape}")
            else:
                # 这种情况通常是 Stage 1 的 checkpoint 里没有 classifier，正常现象
                pass

        # 4. 加载权重 (Strict=False 是核心，允许 classifier 层没有权重)
        msg = self.load_state_dict(new_state_dict, strict=False)
        print(f"Load status: {msg}")
        print("Note: 'Missing keys' for classifier is EXPECTED in Stage 2 fine-tuning.")


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