import os
import pickle
import numpy as np
import torch

from torch.utils.data import DataLoader

from data_provider.data_loader import (
    Dataset_MHealth,
    Dataset_CAPTURE24,
    Dataset_USCHAD,
    Dataset_UCIHAR_Official,
    Dataset_PAMAP50,
    Dataset_PAMAP,
    Dataset_WISDM,
    Dataset_HHAR_1user,
    Dataset_HHAR_cross_user,
    Dataset_MotionSense,
)

from data_provider.mhealth import collate_fn_mhealth
from data_provider.uea import collate_fn

# ============================================================
# Optional SensorLLM full dataset
# ============================================================
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

try:
    from data_provider.data_sensorllm_full import (
        SensorLLMFullDataset,
        SensorLLMFullCollator,
    )
except Exception:
    SensorLLMFullDataset = None
    SensorLLMFullCollator = None


# ============================================================
# Dataset registry
# ============================================================

data_dict = {
    "uschad": Dataset_USCHAD,
    "ucihar": Dataset_UCIHAR_Official,
    "pamap50": Dataset_PAMAP50,
    "pamap": Dataset_PAMAP,
    "capture24": Dataset_CAPTURE24,
    "mhealth": Dataset_MHealth,

    "USCHAD": Dataset_USCHAD,
    "UCIHAR": Dataset_UCIHAR_Official,
    "PAMAP50": Dataset_PAMAP50,
    "PAMAP": Dataset_PAMAP,
    "CAPTURE24": Dataset_CAPTURE24,
    "MHealth": Dataset_MHealth,

    "WISDM": Dataset_WISDM,
    "wisdm": Dataset_WISDM,

    "HHAR_1user": Dataset_HHAR_1user,
    "hhar_1user": Dataset_HHAR_1user,

    "HHAR_cross_user": Dataset_HHAR_cross_user,
    "hhar_cross_user": Dataset_HHAR_cross_user,

    "MotionSense": Dataset_MotionSense,
    "motionsense": Dataset_MotionSense,
}

HHAR_DATASETS = [
    "MHealth",
    "mhealth",
    "USCHAD",
    "uschad",
    "UCIHAR",
    "ucihar",
    "PAMAP50",
    "pamap50",
    "PAMAP",
    "pamap",
    "CAPTURE24",
    "capture24",
    "WISDM",
    "wisdm",
    "HHAR_1user",
    "hhar_1user",
    "HHAR_cross_user",
    "hhar_cross_user",
    "MotionSense",
    "motionsense",
]


# ============================================================
# Generic helpers
# ============================================================

def _flag_is_train(flag):
    return str(flag).upper() in ["TRAIN", "TRAINING"]


def _flag_is_test(flag):
    return str(flag).upper() in ["TEST", "EVAL", "VAL", "VALID", "VALIDATION"]


def _normalize_flag_for_cache(flag):
    if _flag_is_train(flag):
        return "train"
    return "test"


def _get_dataset_key(args):
    return str(getattr(args, "dataset_key", getattr(args, "data", "data"))).lower()


def _get_label_names(args):
    if hasattr(args, "label_names") and args.label_names is not None:
        return [str(x) for x in args.label_names]

    ds_cfg = getattr(args, "ds_cfg", None)

    if isinstance(ds_cfg, dict):
        if "label_names" in ds_cfg and ds_cfg["label_names"] is not None:
            return [str(x) for x in ds_cfg["label_names"]]

        if "id2label" in ds_cfg and ds_cfg["id2label"] is not None:
            id2label = {int(k): str(v) for k, v in ds_cfg["id2label"].items()}
            num_class = int(ds_cfg.get("num_labels", len(id2label)))
            return [id2label[i] for i in range(num_class)]

    num_class = int(getattr(args, "num_class", getattr(args, "num_labels", 0)))
    return [f"class_{i}" for i in range(num_class)]


def _get_channel_names(args):
    if hasattr(args, "channel_names") and args.channel_names is not None:
        return [str(x) for x in args.channel_names]

    ds_cfg = getattr(args, "ds_cfg", None)
    channel_num = int(getattr(args, "enc_in", 0))

    if isinstance(ds_cfg, dict):
        if "channel_names" in ds_cfg and ds_cfg["channel_names"] is not None:
            names = [str(x) for x in ds_cfg["channel_names"]]
            return names

        if "channel_num" in ds_cfg:
            channel_num = int(ds_cfg["channel_num"])

    return [f"channel_{i}" for i in range(channel_num)]


def _get_sample_rate(args):
    if hasattr(args, "sample_rate"):
        return int(args.sample_rate)

    ds_cfg = getattr(args, "ds_cfg", None)

    if isinstance(ds_cfg, dict) and "sample_rate" in ds_cfg:
        return int(ds_cfg["sample_rate"])

    return 50


def _resolve_raw_dataset_name(args):
    """
    SensorLLMFullAdapter 可以有两种用法：

    1. args.data 仍然是 UCIHAR / MHealth / PAMAP
       --model SensorLLMFullAdapter

    2. args.data 写成 SensorLLM_UCIHAR
       --sensorllm_base_data UCIHAR

    这里统一解析到底层原始 HAR 数据集名称。
    """
    if hasattr(args, "sensorllm_base_data") and args.sensorllm_base_data is not None:
        return str(args.sensorllm_base_data)

    data_name = str(getattr(args, "data", ""))

    lower = data_name.lower()

    if lower.startswith("sensorllm_"):
        return data_name.split("_", 1)[1]

    if lower.startswith("sensorllm-"):
        return data_name.split("-", 1)[1]

    return data_name


def _is_sensorllm_full_mode(args):
    model_name = str(getattr(args, "model", "")).lower()

    if model_name in {
        "sensorllmfulladapter",
        "sensor_llm_full_adapter",
        "sensorllm_full",
        "sensorllmfull",
    }:
        return True

    if bool(int(getattr(args, "use_sensorllm_full_data", 0))):
        return True

    data_name = str(getattr(args, "data", "")).lower()

    if data_name.startswith("sensorllm_") or data_name.startswith("sensorllm-"):
        return True

    return False


def _extract_xy_from_raw_item(item, args):
    """
    Try to extract:
        x: [L, C]
        y: int

    from different HAR dataset item formats.

    Supported common formats:
        (x, label, padding_mask)
        (x, label)
        {"x": ..., "label": ...}
        {"batch_x": ..., "labels": ...}
    """
    if isinstance(item, dict):
        x = item.get("batch_x", item.get("x", item.get("features", None)))
        y = item.get("labels", item.get("label", item.get("y", None)))
    elif isinstance(item, (list, tuple)):
        if len(item) < 2:
            raise ValueError(f"Dataset item tuple/list length < 2: len={len(item)}")
        x, y = item[0], item[1]
    else:
        raise TypeError(f"Unsupported dataset item type: {type(item)}")

    if x is None or y is None:
        raise ValueError("Cannot extract x/y from dataset item.")

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    if torch.is_tensor(y):
        y = y.detach().cpu().view(-1)[0].item()
    elif isinstance(y, np.ndarray):
        y = np.asarray(y).reshape(-1)[0].item()
    elif isinstance(y, (list, tuple)):
        y = y[0]

    y = int(y)

    # Ensure x is [L, C]
    enc_in = int(getattr(args, "enc_in", 0))

    if x.ndim != 2:
        raise ValueError(f"x should be 2D [L,C] or [C,L], got shape={x.shape}")

    if enc_in > 0:
        if x.shape[1] == enc_in:
            pass
        elif x.shape[0] == enc_in:
            x = x.T
        else:
            # fallback: leave as is, but warn by raising clear error
            raise ValueError(
                f"Cannot infer channel dimension for x shape={x.shape}, enc_in={enc_in}. "
                "Expected [L,C] or [C,L]."
            )

    return x.astype(np.float32), y


def _export_raw_dataset_to_sensorllm_cache(raw_dataset, args, data_pkl_path, label_pkl_path):
    """
    Convert your existing HAR dataset to SensorLLM cache files:
        data_pkl:  list of np.ndarray [L, C]
        label_pkl: list of int
    """
    os.makedirs(os.path.dirname(data_pkl_path), exist_ok=True)

    data_list = []
    label_list = []

    print(f"[SensorLLMFullData] exporting raw dataset to cache:")
    print(f"  data_pkl  = {data_pkl_path}")
    print(f"  label_pkl = {label_pkl_path}")
    print(f"  n_raw     = {len(raw_dataset)}")

    for i in range(len(raw_dataset)):
        x, y = _extract_xy_from_raw_item(raw_dataset[i], args)
        data_list.append(x)
        label_list.append(int(y))

        if i < 3:
            print(
                f"[SensorLLMFullData][example {i}] "
                f"x_shape={x.shape}, y={y}"
            )

    with open(data_pkl_path, "wb") as f:
        pickle.dump(data_list, f)

    with open(label_pkl_path, "wb") as f:
        pickle.dump(label_list, f)

    print(f"[SensorLLMFullData] saved n={len(data_list)}")

    return data_list, label_list


def _prepare_tokenizer_for_sensorllm(args, channel_names):
    if AutoTokenizer is None:
        raise ImportError("transformers.AutoTokenizer is required for SensorLLMFullDataset.")

    llm_path = getattr(args, "llama_name", None)

    if llm_path is None:
        raise ValueError("--llama_name is required when using SensorLLMFullAdapter.")

    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        use_fast=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    default_ts_token = str(getattr(args, "default_ts_token", "<ts>"))

    special_tokens = [default_ts_token]

    for ch in channel_names:
        special_tokens.append(f"<{ch}_start>")
        special_tokens.append(f"<{ch}_end>")

    tokenizer.add_special_tokens({
        "additional_special_tokens": list(dict.fromkeys(special_tokens))
    })

    # model_max_length 太短会截断大量 QA prompt，这里给一个安全值。
    prompt_max_len = int(getattr(args, "sensorllm_model_max_length", 2048))
    tokenizer.model_max_length = prompt_max_len

    print(f"[SensorLLMFullData] tokenizer vocab size after special tokens: {len(tokenizer)}")
    print(f"[SensorLLMFullData] default_ts_token={default_ts_token}")
    print(f"[SensorLLMFullData] added special token count={len(special_tokens)}")

    return tokenizer


def _build_sensorllm_full_dataset_and_loader(args, flag):
    """
    Build SensorLLM full data pipeline.

    It uses your existing HAR dataset as raw source, exports pkl cache if needed,
    builds/loads QA JSON, tokenizes QA prompt, and returns dict batch.
    """
    if SensorLLMFullDataset is None or SensorLLMFullCollator is None:
        raise ImportError(
            "Cannot import SensorLLMFullDataset/SensorLLMFullCollator. "
            "Please create data_provider/data_sensorllm_full.py first."
        )

    base_data_name = _resolve_raw_dataset_name(args)

    if base_data_name not in data_dict:
        raise KeyError(
            f"Base dataset '{base_data_name}' not found in data_dict. "
            f"Available: {list(data_dict.keys())}"
        )

    RawData = data_dict[base_data_name]

    split_name = _normalize_flag_for_cache(flag)

    shuffle_flag = False if _flag_is_test(flag) else True
    drop_last = False
    batch_size = int(getattr(args, "batch_size", 32))

    dataset_key = _get_dataset_key(args)
    channel_names = _get_channel_names(args)
    label_names = _get_label_names(args)
    sample_rate = _get_sample_rate(args)

    # Make sure args also carries these for model side consistency.
    args.channel_names = channel_names
    args.label_names = label_names
    args.sample_rate = sample_rate
    args.enc_in = len(channel_names)
    args.num_class = len(label_names)

    cache_root = getattr(
        args,
        "sensorllm_cache_dir",
        os.path.join("./sensorllm_cache", dataset_key),
    )

    os.makedirs(cache_root, exist_ok=True)

    data_pkl_path = getattr(
        args,
        f"sensorllm_{split_name}_data_pkl",
        os.path.join(cache_root, f"{dataset_key}_{split_name}_data.pkl"),
    )

    label_pkl_path = getattr(
        args,
        f"sensorllm_{split_name}_label_pkl",
        os.path.join(cache_root, f"{dataset_key}_{split_name}_label.pkl"),
    )

    qa_path = getattr(
        args,
        f"sensorllm_{split_name}_qa_path",
        os.path.join(cache_root, f"{dataset_key}_{split_name}_qa_stage2_cls.json"),
    )

    force_export = bool(int(getattr(args, "force_build_sensorllm_cache", 0)))

    if (not os.path.exists(data_pkl_path)) or (not os.path.exists(label_pkl_path)) or force_export:
        raw_dataset = RawData(
            args=args,
            root_path=args.root_path,
            flag=flag,
        )

        _export_raw_dataset_to_sensorllm_cache(
            raw_dataset=raw_dataset,
            args=args,
            data_pkl_path=data_pkl_path,
            label_pkl_path=label_pkl_path,
        )
    else:
        print(f"[SensorLLMFullData] use existing cache:")
        print(f"  data_pkl  = {data_pkl_path}")
        print(f"  label_pkl = {label_pkl_path}")

    tokenizer = _prepare_tokenizer_for_sensorllm(args, channel_names)

    preprocess_type = str(
        getattr(args, "sensorllm_preprocess_type", "smry+trend+corr+Q")
    )

    default_ts_token = str(getattr(args, "default_ts_token", "<ts>"))
    add_channel_text = bool(int(getattr(args, "sensorllm_add_channel_text", 1)))
    force_build_qa = bool(int(getattr(args, "force_build_sensorllm_qa", 0)))

    seq_len = int(getattr(args, "seq_len", 128))

    data_set = SensorLLMFullDataset(
        data_path=data_pkl_path,
        label_path=label_pkl_path,
        qa_path=qa_path,
        tokenizer=tokenizer,
        split=split_name,
        dataset_key=dataset_key,
        seq_len=seq_len,
        channel_names=channel_names,
        label_names=label_names,
        sample_rate=sample_rate,
        preprocess_type=preprocess_type,
        force_build_qa=force_build_qa,
        default_ts_token=default_ts_token,
        add_channel_text=add_channel_text,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=int(getattr(args, "num_workers", 0)),
        drop_last=drop_last,
        collate_fn=SensorLLMFullCollator(tokenizer=tokenizer),
    )

    print(
        f"[SensorLLMFullData] flag={flag}, split={split_name}, "
        f"n={len(data_set)}, batch_size={batch_size}, shuffle={shuffle_flag}"
    )

    return data_set, data_loader


# ============================================================
# Main data provider
# ============================================================

def data_provider(args, flag):
    """
    Unified data provider.

    For normal models:
        returns tuple batch through collate_fn_mhealth.

    For SensorLLMFullAdapter:
        returns dict batch:
            {
                batch_x,
                labels,
                input_ids,
                attention_mask,
                input_texts,
                answer
            }
    """

    # ------------------------------------------------------------
    # 0. SensorLLM full path
    # ------------------------------------------------------------
    if _is_sensorllm_full_mode(args):
        return _build_sensorllm_full_dataset_and_loader(args, flag)

    # ------------------------------------------------------------
    # 1. Original path
    # ------------------------------------------------------------
    if args.data not in data_dict:
        raise KeyError(
            f"Unknown data name: {args.data}. "
            f"Available keys: {list(data_dict.keys())}"
        )

    Data = data_dict[args.data]

    timeenc = 0 if args.embed != "timeF" else 1

    shuffle_flag = False if (flag == "test" or flag == "TEST") else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == "anomaly_detection":
        drop_last = False

        data_set = Data(
            args=args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )

        print(flag, len(data_set))

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )

        return data_set, data_loader

    elif (
        args.task_name == "classification"
        or args.task_name == "vqvae"
        or args.task_name == "alignment"
    ):
        if args.data in HHAR_DATASETS:
            drop_last = False

            data_set = Data(
                args=args,
                root_path=args.root_path,
                flag=flag,
            )

            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=collate_fn_mhealth,
            )

            return data_set, data_loader

        elif args.data == "UEA":
            drop_last = False

            data_set = Data(
                args=args,
                root_path=args.root_path,
                flag=flag,
            )

            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),
            )

            return data_set, data_loader

        else:
            raise ValueError(
                f"Unsupported classification/vqvae/alignment data: {args.data}. "
                f"If using SensorLLMFullAdapter, set --model SensorLLMFullAdapter "
                f"or --use_sensorllm_full_data 1."
            )

    else:
        if args.data == "m4":
            drop_last = False

        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
        )

        print(flag, len(data_set))

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )

        return data_set, data_loader