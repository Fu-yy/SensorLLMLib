import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
from typing import Dict, Any


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # ================== 初始化逻辑同上 ==================
        print(f"[DeepConvLSTMModel (Original)] Init...")
        self.stage = int(getattr(args, "stage", 1))

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

        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 15)))
        self.num_class = int(self.ds_cfg.get("num_labels", getattr(args, "num_class", 12)))
        self.seq_len_orig = int(getattr(args, "seq_len", 200))

        # Params
        self.num_filters = int(getattr(args, "num_filters", 64))
        self.filter_size = int(getattr(args, "filter_size", 5))
        self.num_units_lstm = int(getattr(args, "num_units_lstm", 128))
        self.num_layers_lstm = int(getattr(args, "num_layers_lstm", 2))
        self.dropout_val = float(getattr(args, "dropout", 0.5))

        # ================== 网络架构 (baseline_longer_sliding_window.py) ==================
        self.conv2DLayer1 = nn.Conv2d(1, self.num_filters, (self.filter_size, 1), stride=(1, 1))
        self.relu1 = nn.ReLU()
        self.conv2DLayer2 = nn.Conv2d(self.num_filters, self.num_filters, (self.filter_size, 1), stride=(1, 1))
        self.relu2 = nn.ReLU()
        self.conv2DLayer3 = nn.Conv2d(self.num_filters, self.num_filters, (self.filter_size, 1), stride=(1, 1))
        self.relu3 = nn.ReLU()
        self.conv2DLayer4 = nn.Conv2d(self.num_filters, self.num_filters, (self.filter_size, 1), stride=(1, 1))
        self.relu4 = nn.ReLU()

        lstm_input_size = self.num_filters * self.C
        # 原始代码 baseline 中 bidirectional=False
        self.lstm = nn.LSTM(lstm_input_size, self.num_units_lstm, self.num_layers_lstm,
                            bidirectional=False, dropout=self.dropout_val)

        self.dropout = nn.Dropout(self.dropout_val)

        # 原始版本直接全连接，没有 Attention 层
        self.dense_layer = nn.Linear(self.num_units_lstm, self.num_class)

    def initHidden(self, batch_size, device):
        h0 = torch.randn(self.num_layers_lstm, batch_size, self.num_units_lstm).to(device) * 0.08
        c0 = torch.randn(self.num_layers_lstm, batch_size, self.num_units_lstm).to(device) * 0.08
        return (h0, c0)

    def forward(self, x, padding_mask=None, mode=None, labels=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)

            # 1. 卷积
        convout1 = self.relu1(self.conv2DLayer1(x))
        convout2 = self.relu2(self.conv2DLayer2(convout1))
        convout3 = self.relu3(self.conv2DLayer3(convout2))
        convout4 = self.relu4(self.conv2DLayer4(convout3))

        # 2. Reshape for LSTM
        lstm_input = convout4.permute(2, 0, 1, 3)
        seq_len, batch_size, _, _ = lstm_input.size()
        lstm_input = lstm_input.contiguous().view(seq_len, batch_size, -1)

        # 原代码这里貌似没有显式的 dropout(lstm_input)，但在 RCNN init 里定义了 self.dropout
        # baseline_longer_sliding_window.py 注释掉了 self.dropout(lstm_input)
        # 但为了稳健性，如果定义了 dropout 还是加上比较好，或者你可以根据 strict 需求去掉
        # 这里保留以保持和 Attention 版本一致的输入处理
        # lstm_input = self.dropout(lstm_input)

        # 3. LSTM
        output, hidden = self.lstm(lstm_input, self.initHidden(batch_size, x.device))

        # 4. 获取最后一个时间步的输出 (No Attention)
        current = output[-1]

        # 5. 分类
        logits = self.dense_layer(current)
        return logits