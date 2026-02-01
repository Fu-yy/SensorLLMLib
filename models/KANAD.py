import os

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from typing import List, Tuple, Dict, Any, Optional

try:
    import yaml
except Exception:
    yaml = None


class KANADModel(nn.Module):
    def __init__(self, window: int, order: int, *args, **kwargs) -> None:
        super().__init__()
        self.order = order
        self.window = window
        self.channels = 2 * self.order + 1
        self.register_buffer(
            "orders",
            self._create_custom_periodic_cosine(self.window, self.order).unsqueeze(
                0
            ),  # (1, order, window)
        )
        self.out_conv = nn.Conv1d(self.channels, 1, 1, bias=False)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.final_conv = nn.Linear(window, window)

    def forward(self, x: torch.Tensor, return_last: bool = False, *args, **kwargs):
        res = []
        res.append(x.unsqueeze(1))
        ff = torch.concat(
            [self.orders.repeat(x.size(0), 1, 1)]  # type: ignore
            + [torch.cos(order * x.unsqueeze(1)) for order in range(1, self.order + 1)]
            + [x.unsqueeze(1)],
            dim=1,
        )  # batch,self.channel,window
        res.append(ff)
        ff = self.init_conv(ff)
        ff = self.bn1(ff)
        ff = self.act(ff)
        ff = self.inner_conv(ff) + res.pop()
        ff = self.bn2(ff)
        ff = self.act(ff)
        ff = self.out_conv(ff) + res.pop()
        ff = self.bn3(ff)
        ff = self.act(ff)
        ff = self.final_conv(ff)
        if return_last:
            return ff.squeeze(1), ff
        return ff.squeeze(1)

    def _create_custom_periodic_cosine(self, window: int, period) -> torch.Tensor:
        d = len(period) if isinstance(period, list) else period
        pl = period if isinstance(period, list) else [i for i in range(1, period + 1)]
        result = torch.empty(d, window, dtype=torch.float32)
        for i, p in enumerate(pl):
            t = torch.arange(0, 1, 1 / window, dtype=torch.float32) / p * 2 * np.pi
            result[i, :] = torch.cos(t)
        return result


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.order = configs.d_model
        # -------- dataset cfg load --------
        self.dataset_key = str(getattr(configs, "dataset_key", getattr(configs, "data", "mhealth"))).lower()
        self.ds_cfg: Dict[str, Any] = {}
        if hasattr(configs, "ds_cfg") and isinstance(configs.ds_cfg, dict):
            self.ds_cfg = configs.ds_cfg
        else:
            ts_yaml = getattr(configs, "ts_backbone_yaml", None)
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
        self.device = configs.device
        self.C = int(self.ds_cfg.get("channel_num", getattr(configs, "enc_in", 15)))
        self.num_class = int(self.ds_cfg.get("num_labels", getattr(configs, "num_class", 12)))
        # Encoder
        self.enc = KANADModel(window=self.seq_len, order=configs.d_model)

        self.num_patches = int((self.seq_len * self.C - self.patch_len) / self.stride + 1)  # L
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
            self.configs.seq_len * self.C, self.num_class)

    def anomaly_detection(self, x_enc):
        ## reshape the input [B, L, D] to [B * D, L]
        x_input = rearrange(x_enc, "B L D -> (B D) L")
        enc_out = self.enc(x_input)
        # [B * D, L]
        dec_out = rearrange(enc_out, "(B D) L -> B L D", B=x_enc.size(0))
        # [B, L, D]
        # [B, C, T/P, D]
        output = self.dropout(dec_out.flatten(start_dim=1))
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            raise NotImplementedError(
                "Task forecasting for KANAD is temporarily not supported"
            )
        if self.task_name == "imputation":
            raise NotImplementedError(
                "Task imputation for KANAD is temporarily not supported"
            )
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
            # raise NotImplementedError(
            #     "Task classification for KANAD is temporarily not supported"
            # )
        return None
