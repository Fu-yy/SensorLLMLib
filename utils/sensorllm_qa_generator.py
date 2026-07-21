# utils/sensorllm_qa_generator.py

import os
import json
import random
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_Q_TEMPLATES = [
    "Given the {data_name} from {channel_num} sensor channels, identify the human activity.",
    "Analyze the {data_name} collected from {channel_num} sensor channels and infer the activity.",
    "Please determine the activity type based on the {data_name} from {channel_num} wearable sensor channels.",
]

DEFAULT_TREND_WORDS = {
    "increase": ["increasing", "rising", "upward"],
    "decrease": ["decreasing", "falling", "downward"],
    "steady": ["steady", "stable", "constant"],
}


def _safe_float(x, digits=4):
    try:
        return round(float(x), digits)
    except Exception:
        return 0.0


def _trend_of_array(x: np.ndarray):
    if len(x) < 2:
        return "steady"

    diff = float(x[-1] - x[0])
    std = float(np.std(x)) + 1e-8

    if abs(diff) < 0.1 * std:
        return "steady"
    return "increase" if diff > 0 else "decrease"


def _split_trend_segments(x: np.ndarray, sample_rate: int, max_segments: int = 4):
    n = len(x)
    if n == 0:
        return []

    seg_len = max(1, n // max_segments)
    rows = []

    for i in range(0, n, seg_len):
        j = min(i + seg_len, n)
        if j <= i:
            continue

        seg = x[i:j]
        trend = _trend_of_array(seg)

        rows.append({
            "start_time": _safe_float(i / sample_rate, 2),
            "end_time": _safe_float(j / sample_rate, 2),
            "trend": trend,
        })

    return rows


def _describe_channel(x: np.ndarray, name: str, sample_rate: int):
    mean = _safe_float(np.mean(x), 4)
    std = _safe_float(np.std(x), 4)
    min_v = _safe_float(np.min(x), 4)
    max_v = _safe_float(np.max(x), 4)
    energy = _safe_float(np.mean(x ** 2), 4)

    trend_segments = _split_trend_segments(x, sample_rate)
    trend_sentences = []

    for seg in trend_segments:
        trend_sentences.append(
            f"From {seg['start_time']}s to {seg['end_time']}s, "
            f"the {name} channel shows a {seg['trend']} trend."
        )

    smry = (
        f"The {name} channel has mean {mean}, standard deviation {std}, "
        f"minimum {min_v}, maximum {max_v}, and average energy {energy}."
    )

    trend_text = " ".join(trend_sentences)

    return smry, trend_text


def _correlation_text(data: np.ndarray, channel_names: List[str]):
    if data.shape[1] <= 1:
        return ""

    df = pd.DataFrame(data, columns=channel_names)
    corr = df.corr().fillna(0.0)

    lines = ["Pearson correlation analysis across sensor channels:"]

    for i in range(len(channel_names)):
        for j in range(i + 1, len(channel_names)):
            v = float(corr.iloc[i, j])

            if v >= 0.7:
                desc = "strongly positively correlated"
            elif v >= 0.3:
                desc = "moderately positively correlated"
            elif v >= 0.1:
                desc = "weakly positively correlated"
            elif v <= -0.7:
                desc = "strongly negatively correlated"
            elif v <= -0.3:
                desc = "moderately negatively correlated"
            elif v <= -0.1:
                desc = "weakly negatively correlated"
            else:
                desc = "not significantly correlated"

            lines.append(
                f"The correlation between {channel_names[i]} and {channel_names[j]} is {desc}."
            )

    return "\n".join(lines)


def generate_qa_for_sample(
    data: np.ndarray,
    label_id: int,
    label_names: List[str],
    channel_names: List[str],
    sample_rate: int,
    q_templates: Optional[List[str]] = None,
):
    q_templates = q_templates or DEFAULT_Q_TEMPLATES

    data_name = random.choice([
        "time-series data",
        "sensor data",
        "normalized time-series data",
        "wearable sensor data",
    ])

    q = random.choice(q_templates).format(
        data_name=data_name,
        channel_num=data.shape[1],
    )

    smry_list = []
    trend_list = []

    for c, name in enumerate(channel_names):
        smry, trend_text = _describe_channel(
            data[:, c].astype(float),
            name=name,
            sample_rate=sample_rate,
        )
        smry_list.append(smry)
        trend_list.append(trend_text)

    corr_text = _correlation_text(data, channel_names)

    info_text = (
        f"The sample contains {data.shape[0]} time steps and {data.shape[1]} sensor channels. "
        f"The sampling rate is {sample_rate}Hz."
    )

    answer = label_names[int(label_id)]

    return {
        "Q": q,
        "smry": " ".join(smry_list),
        "trend_text": " ".join(trend_list),
        "corr_text": corr_text,
        "info_text": info_text,
        "A": answer,
        "label": int(label_id),
    }


def build_sensorllm_qa_json(
    data_list,
    label_list,
    save_path: str,
    label_names: List[str],
    channel_names: List[str],
    sample_rate: int,
    force: bool = False,
):
    if os.path.exists(save_path) and not force:
        print(f"[SensorLLM-QA] exists: {save_path}")
        return save_path

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    qa_dict = {
        "author": "",
        "version": "",
        "date": str(datetime.now().date()),
        "dataset": [],
    }

    for idx, (x, y) in enumerate(zip(data_list, label_list)):
        x = np.asarray(x, dtype=np.float64)

        if isinstance(y, dict):
            label_id = int(y.get("activity", y.get("label", 0)))
        else:
            label_id = int(y)

        if x.ndim != 2:
            raise ValueError(f"sample {idx} should be [L,C], got {x.shape}")

        if x.shape[1] != len(channel_names):
            raise ValueError(
                f"sample {idx} channel mismatch: x.C={x.shape[1]}, channel_names={len(channel_names)}"
            )

        qa_pair = generate_qa_for_sample(
            data=x,
            label_id=label_id,
            label_names=label_names,
            channel_names=channel_names,
            sample_rate=sample_rate,
        )

        qa_dict["dataset"].append({
            "index": idx,
            "qa_pair": qa_pair,
        })

        if idx < 3:
            print(f"[SensorLLM-QA][example {idx}] {qa_pair}")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(qa_dict, f, ensure_ascii=False, indent=2)

    print(f"[SensorLLM-QA] saved: {save_path}, n={len(qa_dict['dataset'])}")
    return save_path