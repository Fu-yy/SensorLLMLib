import os
import yaml
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# 假设你的 VQ-VAE 在这里
from models_new_version_run.VQ_VAE import IMU_VQ_Model

# 头部需要增加这个 import
from peft import PeftModel

# --task_name=classification
# --task_name=lora
# --is_training=1
# --root_path="D:/fuy/MyCode/SensorLLMLib/datasets/MHEALTHDATASET"
# --root_path="D:/fuy/MyCode/SensorLLMLib/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
# --model_id=UCIHAR
# --run_id="alignment_weight"
# --datasets=UCIHAR
# --model="SensorLLMFuy_test_withllm_mae_vqvae"
# --model="SensorLoRA"
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

class SensorLoRAModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device

        # ==========================================
        # 1. 加载 VQ-VAE (用于提取离散 Token)
        # ==========================================
        # -------- dataset cfg load (保持你原来的逻辑) --------
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

        self.vq_net = IMU_VQ_Model(args)

        # 加载 VQ 权重
        vqvae_path = getattr(args, "vqvae_path", None)
        self.qua_path = self.ds_cfg.get(vqvae_path, None)

        if self.qua_path is not None:
            # 兼容处理 path
            ckpt_path = self.qua_path + os.sep + "best_wrapper.pth"
            if os.path.exists(ckpt_path):
                print(f"[SensorLoRA] Loading VQ-VAE from {ckpt_path}")
                vq_net_state_dict = torch.load(ckpt_path, map_location='cpu')
                if 'state_dict' in vq_net_state_dict:
                    vq_net_state_dict = vq_net_state_dict['state_dict']
                self.vq_net.load_state_dict(vq_net_state_dict, strict=False)
            else:
                print(f"[SensorLoRA] Warning: VQ Checkpoint not found at {ckpt_path}")

            self.vq_net.eval().to(self.device)
            # 彻底冻结 VQ-VAE
            for param in self.vq_net.parameters():
                param.requires_grad = False

        # VQ Stride 计算
        self.vq_stride = self.vq_net.stride_t ** self.vq_net.down_t
        self.num_vq_codes = self.vq_net.code_num  # e.g., 512

        # ==========================================
        # 2. 初始化 LLM + Tokenizer (核心修正点)
        # ==========================================
        self.model_name = getattr(args, "llama_name", "meta-llama/Llama-2-7b-hf")

        print(f"[SensorLoRA] Loading Tokenizer from {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Llama 必须这一步

        # 【修正1】使用 special_tokens 防止分词错误，并添加关键的 <PMASK>
        prim_tokens = [f"<P{i:03d}>" for i in range(self.num_vq_codes)] + ["<PMASK>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": prim_tokens})
        print(f"[SensorLoRA] Added {len(prim_tokens)} special tokens (including PMASK).")

        print(f"[SensorLoRA] Loading LLM from {self.model_name}...")
        # 【修正2】去掉 device_map="auto"，手动控制 device，避免 TSLib 冲突
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        ).to(self.device)

        # 训练时必须 resize embedding 否则报错
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm.config.use_cache = False  # 训练时必须关闭 Cache

        # ==========================================
        # 3. 配置 LoRA
        # ==========================================
        lora_r = getattr(args, "lora_r", 8)
        lora_alpha = getattr(args, "lora_alpha", 32)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            # target_modules=["q_proj", "v_proj"]
            # 1. 建议增加 target_modules，Llama 结构通常建议把 MLP 也加上效果更好
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            # 2. 【核心】必须保存并训练 Embedding 和 Head，因为词表变了！
            modules_to_save=["embed_tokens", "lm_head"]
        )
        self.llm = get_peft_model(self.llm, peft_config)

        # 伪代码：请分别在 训练脚本 和 老师加载脚本 中打印这几个值
        # print(f"Vocab Size: {self.tokenizer.vocab_size}")
        # print(f"ID of <P000>: {self.tokenizer.convert_tokens_to_ids('<P000>')}")
        # print(f"ID of <PMASK>: {self.tokenizer.convert_tokens_to_ids('<PMASK>')}")
        self.llm.print_trainable_parameters()



    def _pad_to_multiple(self, x, multiple):
        # 辅助函数：确保输入长度符合 VQ stride
        B, L, C = x.shape
        L_pad = ((L + multiple - 1) // multiple) * multiple
        if L_pad != L:
            pad_len = L_pad - L
            pad = torch.zeros((B, pad_len, C), device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=1)
        return x

    def forward(self, x_imu, padding_mask, context=None,device=None):
        """
        这个 forward 函数完美适配你的 exp 训练循环。
        Input:
            x_imu: [B, Seq_Len, Channel] (原始传感器数据)
        Output:
            loss: scalar (LoRA微调的Causal LM loss)
        """
        B = x_imu.shape[0]

        # 1. 确保数据在设备上
        x_imu = x_imu.to(self.device).float()

        # 2. VQ-VAE 获取离散 Token (Batch转换)
        with torch.no_grad():
            # Pad一下防止长度报错
            x_pad = self._pad_to_multiple(x_imu, self.vq_stride)
            # indices: [B, P] (整数索引 0~511)
            indices = self.vq_net.get_token_ids(x_pad)

            # 这里的 indices 是 tensor，我们需要转成 python list 方便 tokenizer 处理
            # indices_list = indices.cpu().numpy().tolist()
        P = indices.size(1)
        # 2. 构造“填空题” Prompt
        # 随机选择每个样本中的 1 个位置进行 Mask
        mask_pos = torch.randint(0, P, (B,), device=self.device)

        batch_prompts = []
        target_tokens = []  # 记录答案

        # 将 tensor 转 list 稍微慢点但逻辑最清晰，构建 Prompt
        ids_list = indices.cpu().numpy().tolist()
        mask_pos_list = mask_pos.cpu().numpy().tolist()

        for b in range(B):
            seq_str = []
            target_val = ids_list[b][mask_pos_list[b]]

            # 构建序列字符串
            for t in range(P):
                if t == mask_pos_list[b]:
                    seq_str.append("<PMASK>")  # 挖空
                else:
                    seq_str.append(f"<P{ids_list[b][t]:03d}>")

            # 构造 Prompt：这才是让 Teacher 学会推理的关键！
            # 格式：[Instruction] Sequence: ... Answer: [Target]
            prompt = (
                "Analyze the sensor sequence and fill in the mask.\n"
                f"Sequence: {' '.join(seq_str)}\n"
                "Answer: "
            )
            target_token_str = f"<P{target_val:03d}>"

            # 放入列表
            batch_prompts.append(prompt + target_token_str)
            target_tokens.append(target_token_str)
        # 打印第一个样本的 prompt
        print(f"\n[DEBUG TRAIN PROMPT] >>{batch_prompts[0]}<<\n")

        # 3. Tokenizer 编码
        encodings = self.tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=512,  # 适当调大，因为加上了 Prompt
            return_tensors="pt"
        ).to(self.device)

        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask

        # 4. 构建 Labels (关键修正：只计算 Answer 的 Loss)
        # 初始化为 -100 (忽略计算)
        labels = torch.full_like(input_ids, -100)

        # 找到每个序列中 "Answer: " 之后那一个 Token 的位置
        # 简单做法：因为我们把 target 拼在最后了，直接取非 padding 的最后一个 token 作为 label
        # 这种做法假设 tokenizer 不会把 <Pxxx> 拆分 (我们用了 special_tokens，所以安全)

        # 计算每个样本的实际长度 (sum attention_mask)
        seq_lens = attention_mask.sum(dim=1) - 1  # 最后一个 token 的下标

        for b in range(B):
            idx = seq_lens[b]
            # 只有最后一个位置 (Target) 需要计算 Loss
            labels[b, idx] = input_ids[b, idx]

        # 5. LLM Forward
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs.loss

    def save_adapter(self, save_dir):
        """在 exp 结束时调用这个保存权重"""
        self.llm.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)