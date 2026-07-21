import os
import argparse
import yaml
import torch
from types import SimpleNamespace

from data_provider.data_factory import data_provider

# 按你的真实路径修改
from models.IMU_VQ_Model import IMU_VQ_Model
from utils.primitive_profile import build_strong_primitive_profile


def label_names_from_ds_cfg(ds_cfg):
    if "label_names" in ds_cfg and ds_cfg["label_names"] is not None:
        return [str(x) for x in ds_cfg["label_names"]]

    if "id2label" in ds_cfg and ds_cfg["id2label"] is not None:
        id2label = {int(k): str(v) for k, v in ds_cfg["id2label"].items()}
        num_labels = int(ds_cfg.get("num_labels", len(id2label)))

        missing = [i for i in range(num_labels) if i not in id2label]
        if len(missing) > 0:
            raise ValueError(f"id2label missing ids: {missing}")

        return [id2label[i] for i in range(num_labels)]

    num_labels = int(ds_cfg.get("num_labels", 0))
    return [f"class_{i}" for i in range(num_labels)]


def channel_names_from_ds_cfg(ds_cfg):
    channel_num = int(ds_cfg["channel_num"])

    if "channel_names" in ds_cfg and ds_cfg["channel_names"] is not None:
        names = [str(x) for x in ds_cfg["channel_names"]]
        if len(names) != channel_num:
            raise ValueError(
                f"channel_names length mismatch: got {len(names)}, expected {channel_num}"
            )
        return names

    inferred = []
    for key in ds_cfg.keys():
        if key.startswith("default_") and key.endswith("_start_token"):
            name = key.replace("default_", "").replace("_start_token", "")
            inferred.append(name)

    unique = []
    for name in inferred:
        if name not in unique:
            unique.append(name)

    if len(unique) == channel_num:
        return unique

    return [f"channel_{i}" for i in range(channel_num)]


def load_yaml_cfg(ts_backbone_yaml, dataset_key):
    with open(ts_backbone_yaml, "r", encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)

    dataset_key = str(dataset_key).lower()

    if dataset_key not in cfg_all:
        raise KeyError(
            f"dataset_key={dataset_key} not found in {ts_backbone_yaml}. "
            f"Available keys: {list(cfg_all.keys())}"
        )

    return cfg_all[dataset_key]


def resolve_vqvae_path(ds_cfg, vqvae_path):
    if vqvae_path in ds_cfg:
        return ds_cfg[vqvae_path]

    if os.path.exists(vqvae_path):
        return vqvae_path

    raise FileNotFoundError(
        f"Cannot resolve vqvae_path={vqvae_path}. "
        "It should be either a key in yaml, e.g. all_path, or a real checkpoint dir."
    )


def load_vq_model(args, ckpt_dir, device):
    ckpt_path = os.path.join(ckpt_dir, "best_wrapper.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"VQ checkpoint not found: {ckpt_path}")

    model = IMU_VQ_Model(args).to(device)

    sd = torch.load(ckpt_path, map_location="cpu")

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    missing, unexpected = model.load_state_dict(sd, strict=False)

    print(f"[VQ] loaded from: {ckpt_path}")
    print(f"[VQ] missing keys: {missing[:10]}")
    print(f"[VQ] unexpected keys: {unexpected[:10]}")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--dataset_key", type=str, required=True)
    parser.add_argument("--ts_backbone_yaml", type=str, required=True)

    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--patch_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)

    parser.add_argument("--vqvae_path", type=str, default="all_path")
    parser.add_argument("--save_root", type=str, default="./primitive_profiles")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--keep_examples_per_code", type=int, default=5)
    parser.add_argument("--min_valid_ratio_per_token", type=float, default=0.5)

    # Dataset-specific optional args. Keep them for your data_provider compatibility.
    parser.add_argument("--test_subjects", type=str, default=None)
    parser.add_argument("--pamap_variant", type=str, default=None)
    parser.add_argument("--test_users", type=str, default=None)
    parser.add_argument("--val_users", type=str, default=None)
    parser.add_argument("--wisdm_norm", type=str, default="none")
    parser.add_argument("--hhar_tol", type=float, default=0.05)
    parser.add_argument("--hhar_align_on", type=str, default="Arrival_Time")
    parser.add_argument("--hhar_use_cache", type=int, default=1)
    parser.add_argument("--hhar_norm", type=str, default="none")
    parser.add_argument("--motionsense_feature_set", type=str, default="A12")
    parser.add_argument("--motionsense_combine_grav_acc", type=int, default=0)
    parser.add_argument("--motionsense_norm", type=str, default="none")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_cfg = load_yaml_cfg(
        ts_backbone_yaml=args.ts_backbone_yaml,
        dataset_key=args.dataset_key,
    )

    args.ds_cfg = ds_cfg
    args.enc_in = int(ds_cfg["channel_num"])
    args.num_class = int(ds_cfg["num_labels"])
    args.sample_rate = int(ds_cfg.get("sample_rate", 0))

    # Required by IMU_VQ_Model
    args.device = device
    args.d_model = 512
    args.down_sampling_layers = 3
    args.loss_style = "all"

    label_names = label_names_from_ds_cfg(ds_cfg)
    channel_names = channel_names_from_ds_cfg(ds_cfg)

    args.label_names = label_names
    args.channel_names = channel_names

    print(f"[Profile] dataset_key={args.dataset_key}")
    print(f"[Profile] label_names={label_names}")
    print(f"[Profile] channel_names={channel_names}")

    vq_ckpt_dir = resolve_vqvae_path(ds_cfg, args.vqvae_path)
    vq_ckpt_dir = os.path.abspath(vq_ckpt_dir)

    vq_model = load_vq_model(
        args=args,
        ckpt_dir=vq_ckpt_dir,
        device=device,
    )

    _, train_loader = data_provider(args, flag="TRAIN")

    save_dir = os.path.join(args.save_root, str(args.dataset_key).lower())
    os.makedirs(save_dir, exist_ok=True)

    with_label_path = os.path.join(save_dir, "primitive_profile_with_label.json")
    no_label_path = os.path.join(save_dir, "primitive_profile_no_label.json")

    print(f"[Profile] building with-label profile -> {with_label_path}")

    build_strong_primitive_profile(
        vq_model=vq_model,
        dataloader=train_loader,
        save_path=with_label_path,
        device=device,
        num_codes=int(getattr(vq_model, "code_num", 512)),
        label_names=label_names,
        channel_names=channel_names,
        max_batches=args.max_batches,
        keep_examples_per_code=args.keep_examples_per_code,
        min_valid_ratio_per_token=args.min_valid_ratio_per_token,
        include_meta=True,
        include_label_association=True,
    )

    print(f"[Profile] building no-label profile -> {no_label_path}")

    build_strong_primitive_profile(
        vq_model=vq_model,
        dataloader=train_loader,
        save_path=no_label_path,
        device=device,
        num_codes=int(getattr(vq_model, "code_num", 512)),
        label_names=label_names,
        channel_names=channel_names,
        max_batches=args.max_batches,
        keep_examples_per_code=args.keep_examples_per_code,
        min_valid_ratio_per_token=args.min_valid_ratio_per_token,
        include_meta=True,
        include_label_association=False,
    )

    print("=" * 80)
    print("[Done] Primitive profiles generated.")
    print(f"[with-label] {with_label_path}")
    print(f"[no-label]   {no_label_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()