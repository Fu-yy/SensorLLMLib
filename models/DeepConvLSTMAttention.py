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
        return logits