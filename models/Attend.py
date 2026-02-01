import torch
import torch.nn as nn
import os
import yaml
from typing import Dict, Any

# 假设这些工具函数在你的 utils 包中可用，保持引用
# 如果没有这些文件，你需要确保 SelfAttention 和 TemporalAttention 的代码在某处定义
from utils.utils_attention import SelfAttention, TemporalAttention


# ==========================================
# 保持原有的 FeatureExtractor 和 Classifier 不变
# ==========================================

class FeatureExtractor(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            filter_num,
            filter_size,
            enc_num_layers,
            enc_is_bidirectional,
            dropout,
            dropout_rnn,
            activation,
            sa_div,
    ):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(1, filter_num, (filter_size, 1))
        self.conv2 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv3 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv4 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.activation = nn.ReLU() if activation == "ReLU" else nn.Tanh()

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            filter_num * input_dim,
            hidden_dim,
            enc_num_layers,
            bidirectional=enc_is_bidirectional,
            dropout=dropout_rnn,
        )

        self.ta = TemporalAttention(hidden_dim)
        self.sa = SelfAttention(filter_num, sa_div)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Channels]
        x = x.unsqueeze(1)  # -> [Batch, 1, Seq_Len, Channels] for Conv2d

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        # apply self-attention on each temporal dimension (along sensor and feature dimensions)
        # x shape after conv: [Batch, FilterNum, NewSeqLen, Channels]
        refined = torch.cat(
            [self.sa(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
            dim=-1,
        )
        x = refined.permute(3, 0, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)
        outputs, h = self.rnn(x)

        # apply temporal attention on GRU outputs
        out = self.ta(outputs)
        return out


class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_class):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, z):
        return self.fc(z)


# ==========================================
# 适配后的 Model 类
# ==========================================

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # -------------------------------------------------------
        # 1. 解析基础配置 (保留你提供的逻辑)
        # -------------------------------------------------------
        self.stage = int(getattr(args, "stage", 1))
        self.device = args.device
        print(f"[Model] Init in Stage: {self.stage}")

        # Dataset cfg load
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

        # 映射核心维度参数
        # input_dim -> self.C
        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 15)))
        # num_class -> self.num_class
        self.num_class = int(self.ds_cfg.get("num_labels", getattr(args, "num_class", 12)))
        self.seq_len_orig = int(getattr(args, "seq_len", 200))

        # -------------------------------------------------------
        # 2. 提取 ATTEND 模型所需的超参数 (从 args 中读取，没有则使用默认值)
        # -------------------------------------------------------
        hidden_dim = int(getattr(args, "hidden_dim", 128))
        filter_num = int(getattr(args, "filter_num", 64))
        filter_size = int(getattr(args, "filter_size", 5))
        enc_num_layers = int(getattr(args, "enc_num_layers", 2))
        enc_is_bidirectional = bool(getattr(args, "enc_is_bidirectional", False))
        dropout = float(getattr(args, "dropout", 0.5))
        dropout_rnn = float(getattr(args, "dropout_rnn", 0.25))
        dropout_cls = float(getattr(args, "dropout_cls", 0.5))
        activation = str(getattr(args, "activation", "ReLU"))
        sa_div = int(getattr(args, "sa_div", 1))

        print(f"[Model] Building AttendDiscriminate with C={self.C}, NumClass={self.num_class}")

        # -------------------------------------------------------
        # 3. 初始化子模块
        # -------------------------------------------------------
        self.fe = FeatureExtractor(
            input_dim=self.C,
            hidden_dim=hidden_dim,
            filter_num=filter_num,
            filter_size=filter_size,
            enc_num_layers=enc_num_layers,
            enc_is_bidirectional=enc_is_bidirectional,
            dropout=dropout,
            dropout_rnn=dropout_rnn,
            activation=activation,
            sa_div=sa_div,
        )

        self.dropout = nn.Dropout(dropout_cls)
        self.classifier = Classifier(hidden_dim, self.num_class)

        # 注册 buffer，通常用于 metric learning 或 loss 计算
        # 注意：如果这只是用于占位，可以保留；如果不需要 center loss，可以移除
        self.register_buffer(
            "centers", (torch.randn(self.num_class, hidden_dim))
        )

    def forward(self, x,padding_mask=None, mode=None, labels=None):
        """
        x shape: [Batch, Seq_Len, Channels]
        """
        # 特征提取
        feature = self.fe(x)

        # 归一化特征 (z)，通常用于对比学习或特征可视化
        z = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature)
        )

        # 分类
        out = self.dropout(feature)
        logits = self.classifier(out)

        # 返回特征和 logits，保持原逻辑
        return logits