import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Model
import os
import yaml
from typing import Dict, Any

# 假设 DataEmbedding 在 layers.Embed 中，保持引用
# 如果你的环境中没有这个特定的 Embedding 层，可以用 nn.Linear 替代
from layers.Embed import DataEmbedding


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # =======================================================
        # 1. 解析基础配置 (你的框架逻辑)
        # =======================================================
        self.stage = int(getattr(args, "stage", 1))
        self.device = args.device
        print(f"[Model] Init in Stage: {self.stage}")

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

        # 提取关键维度
        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 15)))  # feat_dim
        self.num_class = int(self.ds_cfg.get("num_labels", getattr(args, "num_class", 12)))
        self.seq_len_orig = int(getattr(args, "seq_len", 200))  # seq_len

        print(f"[Model] Building GPT4TS with C={self.C}, SeqLen={self.seq_len_orig}, NumClass={self.num_class}")

        # =======================================================
        # 2. GPT4TS 特有初始化 (参数映射)
        # =======================================================

        # 从 args 读取超参数，如果没有则使用 GPT4TS 的常用默认值
        self.patch_size = int(getattr(args, 'patch_size', 8))
        self.stride = int(getattr(args, 'stride', 8))
        self.gpt_layers = int(getattr(args, 'gpt_layers', 6))
        self.d_model = int(getattr(args, 'd_model', 768))  # GPT2 base 默认为 768
        dropout = float(getattr(args, 'dropout', 0.1))

        # 计算 Patch 数量
        # 逻辑保持原版：(seq_len - patch_size) // stride + 1
        self.patch_num = (self.seq_len_orig - self.patch_size) // self.stride + 1

        # Padding 层
        # 原代码逻辑：先 padding stride 长度，patch_num + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        # 加载预训练 GPT2
        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        # 注意：这需要网络连接下载模型，或者本地有缓存
        # 截断层数，只用前 gpt_layers 层
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]

        self.d_model = self.gpt2.config.n_embd  # 通常是 768
        # Embedding 层
        # 输入维度是: channel * patch_size (因为 patch 后是展平的)
        self.enc_embedding = DataEmbedding(
            self.C * self.patch_size,
            self.d_model,
            dropout
        )


        # 冻结参数 (PEFT 策略)
        # 只训练 ln (LayerNorm) 和 wpe (位置编码)，其余冻结
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 这里的 device设置移除，依靠 Model 实例化后的 .cuda()

        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)

        # 输出投影层
        # 将所有 patch 的输出拼接后投影到类别空间
        self.ln_proj = nn.LayerNorm(self.d_model * self.patch_num)
        self.out_layer = nn.Linear(self.d_model * self.patch_num, self.num_class)

    def forward(self, x, padding_mask=None, mode=None, labels=None):
        """
        x shape: [Batch, Seq_Len, Channels]
        x_mark_enc: 时间戳特征，GPT4TS 原代码未直接使用，保留参数位
        """
        B, L, M = x.shape

        # GPT4TS Patching 逻辑
        # 1. 调整维度为 [Batch, Channels, Seq_Len] 以进行 unfold
        input_x = rearrange(x, 'b l m -> b m l')

        # 2. Padding
        input_x = self.padding_patch_layer(input_x)

        # 3. Unfold (滑窗切片)
        # shape: [Batch, Channels, Patch_Num, Patch_Size]
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)

        # 4. 重排为 Embedding 输入格式
        # shape: [Batch, Patch_Num, (Channels * Patch_Size)]
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')

        # 5. Embedding
        outputs = self.enc_embedding(input_x, None)

        # 6. GPT2 Forward
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        # 7. Classification Head
        # 展平所有 token 的输出: [Batch, Patch_Num * d_model]
        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)

        return outputs