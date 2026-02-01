import os
import yaml
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# 假设你的 VQ-VAE 在这里
from models_new_version_run.VQ_VAE import IMU_VQ_Model


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
        # 2. 初始化 LLM + Tokenizer
        # ==========================================
        self.model_name = getattr(args, "llama_name", "meta-llama/Llama-2-7b-hf")

        print(f"[SensorLoRA] Loading Tokenizer from {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        # Llama通常没有 pad_token，将其设为 eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 添加 <P000> ~ <P511> 到词表
        new_tokens = [f"<P{i:03d}>" for i in range(self.num_vq_codes)]
        num_added = self.tokenizer.add_tokens(new_tokens)
        print(f"[SensorLoRA] Added {num_added} sensor tokens.")

        print(f"[SensorLoRA] Loading LLM from {self.model_name}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"  # 自动分布到显卡
        )
        # 扩展 Embedding 层大小
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # ==========================================
        # 3. 配置 LoRA
        # ==========================================
        print("[SensorLoRA] Applying LoRA Config...")
        lora_r = getattr(args, "lora_r", 8)
        lora_alpha = getattr(args, "lora_alpha", 32)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        self.llm = get_peft_model(self.llm, peft_config)
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
            indices_list = indices.cpu().numpy().tolist()

        # 3. 构造文本 Prompt: "<P001> <P123> <P099> ..."
        batch_texts = []
        for seq_indices in indices_list:
            # 将整数索引转成特殊 Token 字符串
            token_str_list = [f"<P{idx:03d}>" for idx in seq_indices]
            # 拼接成一个字符串
            text = " ".join(token_str_list)
            batch_texts.append(text)

        # 4. Tokenizer 编码 -> input_ids
        # padding=True: 自动补齐 batch 内长度
        # return_tensors='pt': 返回 pytorch tensor
        encodings = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=1024,  # 防止超长
            return_tensors="pt"
        ).to(self.device)

        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask

        # 5. LLM Forward (Causal LM Task)
        # HuggingFace 的模型，只要传入 labels，就会自动计算 loss
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids  # 自监督学习：预测下一个 token，labels 就是 input_ids
        )

        # 直接返回 loss，你的 exp 代码就可以直接 backward 了
        return outputs.loss

    def save_adapter(self, save_dir):
        """在 exp 结束时调用这个保存权重"""
        self.llm.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)