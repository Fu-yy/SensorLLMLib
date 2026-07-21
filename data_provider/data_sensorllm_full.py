# data_provider/data_sensorllm_full.py

import os
import json
import pickle
import copy
from typing import Dict, Sequence, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
import transformers

from utils.sensorllm_qa_generator import build_sensorllm_qa_json

IGNORE_INDEX = -100


def generate_chat_template(messages, bos_token, eos_token, add_generation_prompt=False):
    from jinja2 import Template

    template = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}"
        "{% set content = bos_token + content %}"
        "{% endif %}"
        "{{ content }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
        "{% endif %}"
    )

    return Template(template).render(
        messages=messages,
        bos_token=bos_token or "",
        eos_token=eos_token or "",
        add_generation_prompt=add_generation_prompt,
    )


def _tokenize_fn(conversations: Sequence[str], tokenizer):
    tokenized_list = [
        tokenizer(
            conv,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for conv in conversations
    ]

    input_ids = [x.input_ids[0] for x in tokenized_list]
    input_ids_lens = [
        x.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for x in tokenized_list
    ]

    return {
        "input_ids": input_ids,
        "input_ids_lens": input_ids_lens,
    }


def build_added_str(dataset_key, channel_names, seq_len, default_ts_token="<ts>", add_channel_text=True):
    lines = []

    for ch in channel_names:
        start = f"<{ch}_start>"
        end = f"<{ch}_end>"

        if add_channel_text:
            prefix = f"{ch} readings: "
        else:
            prefix = ""

        lines.append(prefix + start + (default_ts_token * seq_len) + end)

    return "\n".join(lines) + "\n"


def preprocess_cls(
    sources,
    tokenizer,
    added_str,
    preprocess_type,
):
    if preprocess_type == "smry":
        inputs = [added_str + "\n" + s["smry"] for s in sources]
    elif preprocess_type == "trend":
        inputs = [added_str + "\n" + s["trend_text"] for s in sources]
    elif preprocess_type == "corr":
        inputs = [added_str + "\n" + s["corr_text"] for s in sources]
    elif preprocess_type == "none":
        inputs = [added_str for _ in sources]
    elif preprocess_type == "smry+Q":
        inputs = [added_str + "\n" + s["smry"] + "\n" + s["Q"] for s in sources]
    elif preprocess_type == "smry+meta":
        inputs = [added_str + "\n" + s["smry"] + "\n" + s.get("info_text", "") for s in sources]
    elif preprocess_type == "smry+meta+Q":
        inputs = [added_str + "\n" + s["smry"] + "\n" + s.get("info_text", "") + "\n" + s["Q"] for s in sources]
    elif preprocess_type == "smry+corr":
        inputs = [added_str + "\n" + s["smry"] + "\n" + s["corr_text"] for s in sources]
    elif preprocess_type == "smry+corr+Q":
        inputs = [added_str + "\n" + s["smry"] + "\n" + s["corr_text"] + "\n" + s["Q"] for s in sources]
    elif preprocess_type == "smry+trend+corr":
        inputs = [added_str + "\n" + s["smry"] + "\n" + s["trend_text"] + "\n" + s["corr_text"] for s in sources]
    elif preprocess_type == "smry+trend+corr+Q":
        inputs = [added_str + "\n" + s["smry"] + "\n" + s["trend_text"] + "\n" + s["corr_text"] + "\n" + s["Q"] for s in sources]
    elif preprocess_type == "smry+trend+Q":
        inputs = [added_str + "\n" + s["smry"] + "\n" + s["trend_text"] + "\n" + s["Q"] for s in sources]
    else:
        assert preprocess_type == "smry+trend", f"Undefined preprocess_type {preprocess_type}"
        inputs = [added_str + "\n" + s["smry"] + "\n" + s["trend_text"] for s in sources]

    tokenized = _tokenize_fn(inputs, tokenizer)

    return {
        "input_ids": tokenized["input_ids"],
        "input_texts": inputs,
    }


class SensorLLMFullDataset(Dataset):
    def __init__(
        self,
        data_path,
        label_path,
        qa_path,
        tokenizer,
        split,
        dataset_key,
        seq_len,
        channel_names,
        label_names,
        sample_rate,
        preprocess_type="smry+trend+corr+Q",
        force_build_qa=False,
        default_ts_token="<ts>",
        add_channel_text=True,
    ):
        super().__init__()

        self.data_path = data_path
        self.label_path = label_path
        self.qa_path = qa_path
        self.tokenizer = tokenizer
        self.split = split
        self.dataset_key = dataset_key
        self.seq_len = int(seq_len)
        self.channel_names = list(channel_names)
        self.label_names = list(label_names)
        self.sample_rate = int(sample_rate)
        self.preprocess_type = preprocess_type
        self.default_ts_token = default_ts_token
        self.add_channel_text = bool(add_channel_text)

        with open(self.data_path, "rb") as f:
            self.data_file = pickle.load(f)

        with open(self.label_path, "rb") as f:
            self.label_file = pickle.load(f)

        build_sensorllm_qa_json(
            data_list=self.data_file,
            label_list=self.label_file,
            save_path=self.qa_path,
            label_names=self.label_names,
            channel_names=self.channel_names,
            sample_rate=self.sample_rate,
            force=force_build_qa,
        )

        with open(self.qa_path, "r", encoding="utf-8") as f:
            qa_file = json.load(f)

        self.qa_list = qa_file["dataset"]

        if len(self.data_file) != len(self.qa_list):
            raise ValueError(
                f"data and qa mismatch: data={len(self.data_file)}, qa={len(self.qa_list)}"
            )

        self.added_str = build_added_str(
            dataset_key=self.dataset_key,
            channel_names=self.channel_names,
            seq_len=self.seq_len,
            default_ts_token=self.default_ts_token,
            add_channel_text=self.add_channel_text,
        )

        print(
            f"[SensorLLMFullDataset] split={split}, n={len(self.data_file)}, "
            f"seq_len={self.seq_len}, C={len(self.channel_names)}, preprocess_type={self.preprocess_type}"
        )
        print(f"[SensorLLMFullDataset] qa_path={self.qa_path}")
        print(f"[SensorLLMFullDataset] added_str example:\n{self.added_str[:500]}")

    def __len__(self):
        return len(self.data_file)

    def _get_label(self, item):
        if isinstance(item, dict):
            return int(item.get("activity", item.get("label", 0)))
        return int(item)

    def __getitem__(self, index):
        x = np.asarray(self.data_file[index], dtype=np.float32)

        if x.ndim != 2:
            raise ValueError(f"x should be [L,C], got {x.shape}")

        if x.shape[0] != self.seq_len:
            if x.shape[0] > self.seq_len:
                x = x[:self.seq_len]
            else:
                pad = np.zeros((self.seq_len - x.shape[0], x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=0)

        y = self._get_label(self.label_file[index])

        qa_pair = copy.deepcopy(self.qa_list[index]["qa_pair"])
        qa_pair["label"] = y

        data_dict = preprocess_cls(
            [qa_pair],
            tokenizer=self.tokenizer,
            added_str=self.added_str,
            preprocess_type=self.preprocess_type,
        )

        return {
            "batch_x": torch.from_numpy(x).float(),
            "labels": torch.tensor(y).long(),
            "input_ids": data_dict["input_ids"][0],
            "input_texts": data_dict["input_texts"][0],
            "answer": qa_pair["A"],
        }


@dataclass
class SensorLLMFullCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        batch_x = torch.stack([x["batch_x"] for x in instances], dim=0)
        labels = torch.stack([x["labels"] for x in instances], dim=0)

        input_ids = [x["input_ids"] for x in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        return {
            "batch_x": batch_x,
            "labels": labels,
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "input_texts": [x["input_texts"] for x in instances],
            "answer": [x["answer"] for x in instances],
        }