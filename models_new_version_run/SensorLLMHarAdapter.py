# models_new_version_run/SensorLLMFullAdapter.py

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml
except Exception:
    yaml = None

from transformers import AutoTokenizer, AutoModelForCausalLM


def get_label_names_from_cfg(ds_cfg: Dict[str, Any], num_class: int):
    if "label_names" in ds_cfg and ds_cfg["label_names"] is not None:
        return [str(x) for x in ds_cfg["label_names"]]
    if "id2label" in ds_cfg and ds_cfg["id2label"] is not None:
        id2label = {int(k): str(v) for k, v in ds_cfg["id2label"].items()}
        return [id2label[i] for i in range(num_class)]
    return [f"class_{i}" for i in range(num_class)]


def get_channel_names_from_cfg(ds_cfg: Dict[str, Any], channel_num: int):
    if "channel_names" in ds_cfg and ds_cfg["channel_names"] is not None:
        names = [str(x) for x in ds_cfg["channel_names"]]
        if len(names) == channel_num:
            return names
    return [f"channel_{i}" for i in range(channel_num)]


class SimpleChannelTokenEncoder(nn.Module):
    def __init__(self, seq_len, channel_num, llm_hidden, dropout=0.1):
        super().__init__()
        self.seq_len = int(seq_len)
        self.channel_num = int(channel_num)
        self.llm_hidden = int(llm_hidden)

        self.value_proj = nn.Sequential(
            nn.LayerNorm(self.seq_len),
            nn.Linear(self.seq_len, llm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_hidden, llm_hidden),
        )

        self.stat_proj = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, llm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_hidden, llm_hidden),
        )

        self.channel_embed = nn.Parameter(torch.randn(1, channel_num, llm_hidden) * 0.02)

    def forward(self, x):
        # x: [B,L,C]
        B, L, C = x.shape
        if C != self.channel_num:
            raise ValueError(f"Expected C={self.channel_num}, got {C}")

        if L != self.seq_len:
            if L > self.seq_len:
                x = x[:, :self.seq_len, :]
            else:
                pad = x.new_zeros(B, self.seq_len - L, C)
                x = torch.cat([x, pad], dim=1)

        xc = x.transpose(1, 2).contiguous()  # [B,C,L]

        mean = xc.mean(dim=-1, keepdim=True)
        std = xc.std(dim=-1, keepdim=True).clamp_min(1e-5)
        x_norm = (xc - mean) / std

        value_token = self.value_proj(x_norm)

        ch_mean = xc.mean(dim=-1)
        ch_std = xc.std(dim=-1)
        ch_min = xc.min(dim=-1).values
        ch_max = xc.max(dim=-1).values
        ch_energy = (xc ** 2).mean(dim=-1)
        ch_abs = xc.abs().mean(dim=-1)

        stats = torch.stack([ch_mean, ch_std, ch_min, ch_max, ch_energy, ch_abs], dim=-1)
        stat_token = self.stat_proj(stats)

        return value_token + stat_token + self.channel_embed


class AlignmentModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device = args.device

        self.dataset_key = str(getattr(args, "dataset_key", getattr(args, "data", "uci"))).lower()

        if hasattr(args, "ds_cfg") and isinstance(args.ds_cfg, dict):
            self.ds_cfg = args.ds_cfg
        else:
            self.ds_cfg = self._load_ds_cfg_from_yaml(args)

        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 6)))
        self.num_class = int(self.ds_cfg.get("num_labels", getattr(args, "num_class", 6)))
        self.seq_len = int(getattr(args, "seq_len", 128))

        self.label_names = get_label_names_from_cfg(self.ds_cfg, self.num_class)
        self.channel_names = get_channel_names_from_cfg(self.ds_cfg, self.C)

        llm_path = getattr(args, "llama_name", None)
        if llm_path is None:
            raise ValueError("--llama_name is required.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            use_fast=False,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.default_ts_token = str(getattr(args, "default_ts_token", "<ts>"))

        special_tokens = [self.default_ts_token]
        for ch in self.channel_names:
            special_tokens.append(f"<{ch}_start>")
            special_tokens.append(f"<{ch}_end>")

        self.tokenizer.add_special_tokens({
            "additional_special_tokens": list(dict.fromkeys(special_tokens))
        })

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            output_hidden_states=True,
        ).to(self.device)

        self.llm.resize_token_embeddings(len(self.tokenizer))

        for p in self.llm.parameters():
            p.requires_grad = False

        self.llm.eval()

        self.llm_hidden = int(self.llm.config.hidden_size)
        self.llm_dtype = next(self.llm.parameters()).dtype

        self.ts_token_id = self.tokenizer.convert_tokens_to_ids(self.default_ts_token)

        self.start_token_ids = {
            ch: self.tokenizer.convert_tokens_to_ids(f"<{ch}_start>")
            for ch in self.channel_names
        }
        self.end_token_ids = {
            ch: self.tokenizer.convert_tokens_to_ids(f"<{ch}_end>")
            for ch in self.channel_names
        }

        self.sensor_encoder = SimpleChannelTokenEncoder(
            seq_len=self.seq_len,
            channel_num=self.C,
            llm_hidden=self.llm_hidden,
            dropout=float(getattr(args, "dropout", 0.1)),
        )

        self.score = nn.Linear(self.llm_hidden, self.num_class, bias=False)

        self.lambda_lm = float(getattr(args, "lambda_lm", 0.0))

        print(f"[SensorLLMFullAdapter] dataset_key={self.dataset_key}")
        print(f"[SensorLLMFullAdapter] seq_len={self.seq_len}, C={self.C}, num_class={self.num_class}")
        print(f"[SensorLLMFullAdapter] channel_names={self.channel_names}")
        print(f"[SensorLLMFullAdapter] llm_hidden={self.llm_hidden}, dtype={self.llm_dtype}")
        print(f"[SensorLLMFullAdapter] ts_token_id={self.ts_token_id}")
        print(f"[SensorLLMFullAdapter] start_token_ids={self.start_token_ids}")

    def _load_ds_cfg_from_yaml(self, args):
        ts_yaml = getattr(args, "ts_backbone_yaml", None)
        if ts_yaml is None:
            raise ValueError("ts_backbone_yaml is required.")
        if yaml is None:
            raise ImportError("pyyaml is required.")

        if os.path.exists(ts_yaml):
            config_path = ts_yaml
        else:
            project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(project_path, "configs", ts_yaml)

        with open(config_path, "r", encoding="utf-8") as f:
            cfg_all = yaml.safe_load(f)

        if self.dataset_key not in cfg_all:
            raise KeyError(f"{self.dataset_key} not found in {config_path}")

        return cfg_all[self.dataset_key]

    def _replace_ts_embeddings(self, input_ids, input_embeds, sensor_embeds):
        """
        input_ids: [B,T]
        input_embeds: [B,T,H]
        sensor_embeds: [B,C,H]

        For each channel:
            <ch_start> <ts> <ts> ... <ch_end>
        Replace the ts-token span by the corresponding channel embedding repeated
        across all placeholder positions.
        """
        B, T = input_ids.shape
        _, C, H = sensor_embeds.shape

        outputs = []

        for b in range(B):
            cur_ids = input_ids[b]
            cur_emb = input_embeds[b].clone()

            for c, ch in enumerate(self.channel_names):
                st_id = self.start_token_ids[ch]
                ed_id = self.end_token_ids[ch]

                st_pos = torch.where(cur_ids == st_id)[0]
                ed_pos = torch.where(cur_ids == ed_id)[0]

                if len(st_pos) != 1 or len(ed_pos) != 1:
                    continue

                st = int(st_pos[0].item())
                ed = int(ed_pos[0].item())

                if ed <= st + 1:
                    continue

                middle = torch.arange(st + 1, ed, device=cur_ids.device)
                ts_mask = cur_ids[middle] == self.ts_token_id

                if ts_mask.sum() == 0:
                    continue

                ts_positions = middle[ts_mask]

                cur_emb[ts_positions] = sensor_embeds[b, c].unsqueeze(0).expand(len(ts_positions), -1)

            outputs.append(cur_emb)

        return torch.stack(outputs, dim=0)

    def forward(
        self,
        x_imu=None,
        padding_mask=None,
        mode=None,
        labels=None,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        if x_imu is None:
            x_imu = kwargs.get("batch_x", None)

        if x_imu is None:
            raise ValueError("x_imu/batch_x is required.")

        x_imu = x_imu.to(self.device).float()

        if labels is not None:
            labels = labels.to(self.device).long().view(-1)

        if input_ids is None:
            raise ValueError("input_ids is required for SensorLLMFullAdapter.")

        input_ids = input_ids.to(self.device).long()

        if attention_mask is None:
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        else:
            attention_mask = attention_mask.to(self.device).long()

        sensor_embeds = self.sensor_encoder(x_imu)
        sensor_embeds = sensor_embeds.to(dtype=self.llm_dtype)

        input_embeds = self.llm.get_input_embeddings()(input_ids)
        input_embeds = self._replace_ts_embeddings(
            input_ids=input_ids,
            input_embeds=input_embeds,
            sensor_embeds=sensor_embeds,
        )

        self.llm.eval()

        out = self.llm(
            input_ids=None,
            inputs_embeds=input_embeds.to(dtype=self.llm_dtype),
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        hidden = out.hidden_states[-1].float()
        token_logits = self.score(hidden)

        seq_lengths = attention_mask.long().sum(dim=1) - 1
        pooled_logits = token_logits[
            torch.arange(input_ids.shape[0], device=input_ids.device),
            seq_lengths,
        ]

        if labels is None or mode in ["classify", "test", "eval", "eval_no_mask", "inference"]:
            return pooled_logits

        loss_activity = F.cross_entropy(pooled_logits, labels)
        loss_total = loss_activity

        with torch.no_grad():
            pred = pooled_logits.argmax(dim=-1)
            acc = (pred == labels).float().mean()

        metrics = {
            "loss_total": float(loss_total.detach().item()),
            "loss_activity": float(loss_activity.detach().item()),
            "loss_primitive": 0.0,
            "activity_acc": float(acc.detach().item()),
            "primitive_acc": 0.0,
            "mask_ratio_actual": 0.0,
        }

        return loss_total, pooled_logits, metrics

    @torch.no_grad()
    def classify(self, x_imu, padding_mask=None, input_ids=None, attention_mask=None):
        was_training = self.training
        self.eval()

        logits = self.forward(
            x_imu=x_imu,
            padding_mask=padding_mask,
            mode="classify",
            labels=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probs = F.softmax(logits, dim=-1)

        if was_training:
            self.train()

        return logits, probs

    def save_wrapper(self, path):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_wrapper(self, path, map_location="cpu"):
        sd = torch.load(path, map_location=map_location)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        elif isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        elif isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"[SensorLLMFullAdapter][load] missing={missing[:30]}")
        print(f"[SensorLLMFullAdapter][load] unexpected={unexpected[:30]}")
        print(f"[SensorLLMFullAdapter][load] num_missing={len(missing)}, num_unexpected={len(unexpected)}")