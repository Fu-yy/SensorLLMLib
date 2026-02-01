# exp/exp_sensorllm_unified_v2_fixed.py
import copy
import os
import time
import json
import yaml
import warnings
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models_new_version_run.VQ_VAE import validate_vqvae
from utils.tools import EarlyStopping, cal_accuracy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
# 2) collect embeddings from TRAIN
from sklearn.cluster import MiniBatchKMeans
warnings.filterwarnings("ignore")

# [新增] 全局辅助函数：处理 Windows 长路径
import os


'''

2) “聚类不要回滚权重”：正确做法是“临时 load / 提 embedding / 聚类 / 恢复”

你要的论文叙事是：

聚类用一个稳定表征（best_wrapper 或 EMA wrapper），但不改变训练轨迹。

标准流程（必须这么写，reviewer 也认可）

sd_cur = copy(model.state_dict())

model.load_wrapper(best_wrapper)（临时）

extract_embeddings()（提 patch embedding）

kmeans.fit() 得到 centers

model.load_state_dict(sd_cur)（恢复继续训练）

model.set_kmeans_centers(centers)（注入 centers，不改变 backbone 权重）

一个“最小可用”的函数骨架（你照着塞到 Exp 里）

我不替你写全工程，只给你“关键骨架”：

import copy
import torch

@torch.no_grad()
def extract_patch_embeddings_for_kmeans(model, loader, max_batches=50):
    """
    目标：提 z_nomask 的 patch 表征，用于 kmeans
    返回: [N, D] 的二维 tensor (CPU)
    """
    model.eval()
    feats = []

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
        batch_x = batch_x.to(model.device)

        # 走到你模型里 patch_embed 的“无 mask”分支
        x = model._align_seq_len(batch_x)
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True).clamp_min(model.norm_eps)
        x_norm = (x - mu) / sigma

        z = model.patch_embed(x_norm, patch_mask=None)  # [B,P,D] (注意：这里是否含 pos_embed，下面第3节会说)
        feats.append(z.reshape(-1, z.size(-1)).detach().cpu())

    return torch.cat(feats, dim=0)  # [N, D]

def run_kmeans_and_inject_centers(model, train_loader, best_wrapper_path, K=32, max_batches=50):
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    # 1) save current
    sd_cur = copy.deepcopy(model.state_dict())

    # 2) temp load best wrapper
    model.load_wrapper(best_wrapper_path, map_location="cpu")
    model.to(model.device)

    # 3) extract embeddings
    X = extract_patch_embeddings_for_kmeans(model, train_loader, max_batches=max_batches)  # [N,D] CPU
    Xn = X.numpy()

    # 4) kmeans
    km = MiniBatchKMeans(n_clusters=K, batch_size=4096, random_state=0)
    km.fit(Xn)
    centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)  # [K,D]

    # 5) restore
    model.load_state_dict(sd_cur, strict=True)
    model.to(model.device)

    # 6) inject centers
    model.set_kmeans_centers(centers)
    return centers


你会发现：聚类路径用 best_wrapper，但训练参数恢复了，轨迹不会跳变。
这就是你想要的“严谨”。


以上是后期几个阶段的训练策略

'''

def to_secure_path(path):
    """
    处理路径兼容性：
    1. Windows: 转绝对路径 -> 统一反斜杠 -> 加 \\?\ 前缀 (绕过长度限制)
    2. Linux/Mac: 仅转绝对路径 (Linux 无长度限制问题，也不支持 \\?\ 前缀)
    """
    if path is None:
        return None

    # 1. 统一转换为绝对路径 (这一步在 Win/Linux 都会执行)
    secure_path = os.path.abspath(path)

    # 2. 仅针对 Windows 系统做特殊处理
    if os.name == 'nt':
        # [新增] 强制将所有 / 替换为 \，防止 \\?\ 模式下因混合斜杠报错
        secure_path = secure_path.replace('/', '\\')

        # 添加长路径前缀
        if not secure_path.startswith("\\\\?\\"):
            secure_path = "\\\\?\\" + secure_path

    return secure_path
def report_trainable_params(model, topk=200):
    trainable = []
    frozen = []
    for n, p in model.named_parameters():
        (trainable if p.requires_grad else frozen).append((n, p.numel()))
    trainable_sorted = sorted(trainable, key=lambda x: -x[1])
    frozen_sorted = sorted(frozen, key=lambda x: -x[1])

    total = sum(p.numel() for _, p in model.named_parameters())
    trn = sum(x[1] for x in trainable)
    print(f"[Params] total={total/1e6:.2f}M, trainable={trn/1e6:.2f}M ({trn/total*100:.2f}%)")
    print("---- Trainable (top) ----")
    for n, k in trainable_sorted[:topk]:
        print(f"{n:80s} {k/1e6:8.3f}M")
    print("---- Frozen (top) ----")
    for n, k in frozen_sorted[:min(topk, 50)]:
        print(f"{n:80s} {k/1e6:8.3f}M")


def _unwrap(m):
    return m.module if hasattr(m, "module") else m


class TeeLogger:
    def __init__(self, log_path: str, flush: bool = True):
        # [修改] 对 log_path 进行长路径处理
        self.log_path = to_secure_path(log_path)
        self.flush = flush

        # 使用处理后的路径提取目录
        d = os.path.dirname(self.log_path)
        if d:
            os.makedirs(d, exist_ok=True)

    def write(self, msg: str, also_print: bool = True):
        if msg is None:
            return
        msg = str(msg).rstrip("\n")

        if also_print:
            print(msg, flush=self.flush)

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # [修改] self.log_path 已经是处理过的长路径，open 可以正常打开
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
            if self.flush:
                f.flush()


class Exp_VQ_VAE_Classification(Exp_Basic):
    """
    V2-fixed:
      - optimizer param groups (decay/no_decay)
      - scheduler: cosine (epoch-step) + optional warmup
      - early stopping: default monitor val_acc (same behavior as your V1)
    """

    def __init__(self, args):
        self._inject_dataset_cfg(args)
        super().__init__(args)

        # log_root = getattr(self.args, "log_dir", "./logs")
        run_id = getattr(self.args, "run_id", None)
        # if not run_id:
        #     run_id = time.strftime("%Y%m%d_%H%M%S")
        #     setattr(self.args, "run_id", run_id)

        # self.log_dir = os.path.join(log_root, getattr(self.args, "model", "model"), str(run_id))
        # os.makedirs(self.log_dir, exist_ok=True)

        dataset_key = str(getattr(self.args, "dataset_key", getattr(self.args, "data", "data"))).lower()
        self.run_root = os.path.join(getattr(self.args, "run_root", "./runs"),
                                     getattr(self.args, "model", "model"),
                                     dataset_key,
                                     str(run_id))
        os.makedirs(self.run_root, exist_ok=True)

        # self.log_dir = os.path.join(self.run_root, "logs")  # 统一日志
        # os.makedirs(self.log_dir, exist_ok=True)

        self.logger = None

        # stage2 load stage1 if needed
        # if int(getattr(self.args, "stage", 2)) == 2:
        #     # self._maybe_load_stage1_hf(args)
        #     self._maybe_load_stage1_wrapper(args)

        # if bool(getattr(self.args, "freeze_llm", True)):
        #     self.set_trainable_modules()

        self.diag_logits = bool(getattr(self.args, "diag_logits", False))

        # V2 params (defaults)


        # scheduler controls
        if not hasattr(self.args, "use_cosine"):
            setattr(self.args, "use_cosine", True)
        if not hasattr(self.args, "cosine_by_iter"):
            setattr(self.args, "cosine_by_iter", True)  # default iter-step cosine
        if not hasattr(self.args, "monitor"):
            setattr(self.args, "monitor", "acc")         # "acc" or "loss"
        if not hasattr(self.args, "use_class_weight"):
            setattr(self.args, "use_class_weight", True)

        if not hasattr(self.args, "weight_decay"):
            setattr(self.args, "weight_decay", 1e-4)
        if not hasattr(self.args, "min_lr"):
            setattr(self.args, "min_lr", 1e-5)
        if not hasattr(self.args, "warmup_epochs"):
            setattr(self.args, "warmup_epochs", 0)

    def _paths_for_setting(self, setting: str):
        # run_root 已经是 runs/model/dataset/run_id
        p = {}

        # stage1
        p["stage1_log"] = os.path.join(self.run_root, "stage1", "logs", f"{setting}.log")
        p["stage1_ckpt_dir"] = os.path.join(self.run_root, "stage1", "ckpts", setting)
        p["stage1_wrapper"] = os.path.join(p["stage1_ckpt_dir"], "best_wrapper.pth")
        p["stage1_results_dir"] = os.path.join(self.run_root, "stage1", "results", setting)

        # stage2
        p["stage2_log"] = os.path.join(self.run_root, "stage2", "logs", f"{setting}.log")
        p["stage2_ckpt_dir"] = os.path.join(self.run_root, "stage2", "ckpts", setting)
        p["stage2_ckpt"] = os.path.join(p["stage2_ckpt_dir"], "checkpoint.pth")
        p["stage2_results_dir"] = os.path.join(self.run_root, "stage2", "results", setting)

        # meta
        p["meta_dir"] = os.path.join(self.run_root, "meta", setting)
        p["meta_stage1"] = os.path.join(p["meta_dir"], "meta_stage1.json")
        p["meta_stage2"] = os.path.join(p["meta_dir"], "meta_stage2.json")

        return p

    def _dump_meta(self, meta_path: str, stage: str, setting: str, paths: dict):
        meta_path = to_secure_path(meta_path)

        os.makedirs(os.path.dirname(meta_path), exist_ok=True)

        # 只保留 json 可序列化的 args（兜底）
        args_dict = {}
        for k, v in vars(self.args).items():
            try:
                json.dumps(v)
                args_dict[k] = v
            except TypeError:
                args_dict[k] = str(v)

        meta = {
            "stage": stage,  # "stage1" / "stage2"
            "setting": setting,
            "run_id": str(getattr(self.args, "run_id", "")),
            "model": str(getattr(self.args, "model", "")),
            "dataset_key": str(getattr(self.args, "dataset_key", getattr(self.args, "data", ""))),
            "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "paths": paths,  # ✅ 你要的“把路径写进文件”
            "args": args_dict,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # -------------------------
    # config injection
    # -------------------------
    def _inject_dataset_cfg(self, args):
        ts_yaml = getattr(args, "ts_backbone_yaml", None)
        if ts_yaml is None:
            return

        project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(project_path, "configs", ts_yaml)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"ts_backbone_yaml not found: {config_path}")

        dataset_key = str( getattr(args, "data")).lower()
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_all = yaml.safe_load(f)
        if dataset_key not in cfg_all:
            raise KeyError(f"dataset '{dataset_key}' not found in {config_path}")

        ds_cfg = cfg_all[dataset_key]
        args.dataset_key = dataset_key
        args.ds_cfg = ds_cfg

        if "channel_num" in ds_cfg:
            args.enc_in = int(ds_cfg["channel_num"])
        if "sample_rate" in ds_cfg:
            args.sample_rate = int(ds_cfg["sample_rate"])
        if "num_labels" in ds_cfg:
            args.num_class = int(ds_cfg["num_labels"])

    def _is_two_stage_model(self) -> bool:
        if hasattr(self.args, "two_stage"):
            return bool(getattr(self.args, "two_stage"))
        name = str(getattr(self.args, "model", "")).lower()
        m = _unwrap(self.model)
        if "sensorllm" in name:
            return True
        if hasattr(m, "llm") and hasattr(m, "tokenizer"):
            return True
        return False

    # -------------------------
    # build model
    # -------------------------
    def _build_model(self):
        train_data, _ = self._get_data(flag="TRAIN")
        val_data, _ = self._get_data(flag="TEST")
        test_data, _ = self._get_data(flag="TEST")

        self.args.seq_len = max(
            getattr(train_data, "max_seq_len", getattr(self.args, "seq_len", 0)),
            getattr(val_data, "max_seq_len", getattr(self.args, "seq_len", 0)),
            getattr(test_data, "max_seq_len", getattr(self.args, "seq_len", 0)),
        )
        self.args.pred_len = 0

        if hasattr(self.args, "ds_cfg") and isinstance(self.args.ds_cfg, dict):
            self.args.enc_in = int(self.args.ds_cfg.get("channel_num", self.args.enc_in))
            self.args.num_class = int(self.args.ds_cfg.get("num_labels", getattr(self.args, "num_class", 12)))

        model = self.model_dict[self.args.model].IMU_VQ_Model(self.args).float()


        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _autofind_stage1_paths(self, setting: str):
        # save_root = os.path.join(getattr(self.args, "pretrain_checkpoints", "./pretrain_ckpts"), setting)
        save_root = os.path.join(self.run_root, "stage1", "ckpts", setting)

        wrapper_path = os.path.join(save_root, "best_wrapper.pth")
        hf_dir = os.path.join(save_root, "best_hf")  # 你 save_hf_bundle(tag="best") 就是这个

        has_wrapper = os.path.isfile(wrapper_path)
        has_hf = os.path.isdir(hf_dir)

        return save_root, (wrapper_path if has_wrapper else None), (hf_dir if has_hf else None)

    # -------------------------
    # save/load hf
    # -------------------------
    def save_hf_bundle(self, save_dir: str, tag: str = "best"):
        os.makedirs(save_dir, exist_ok=True)
        m = _unwrap(self.model)

        if not (hasattr(m, "llm") and hasattr(m, "tokenizer")):
            raise RuntimeError("save_hf_bundle requires wrapper has .llm and .tokenizer")

        hf_dir = os.path.join(save_dir, f"{tag}_hf")
        os.makedirs(hf_dir, exist_ok=True)

        m.llm.save_pretrained(hf_dir)
        m.tokenizer.save_pretrained(hf_dir)

        meta = {
            "dataset_key": getattr(self.args, "dataset_key", getattr(self.args, "data", None)),
            "ds_cfg": getattr(self.args, "ds_cfg", None),
            "C": getattr(m, "C", getattr(self.args, "enc_in", None)),
            "ts_tokens": getattr(m, "ts_tokens", None),
            "stage": int(getattr(self.args, "stage", 2)),
        }
        with open(os.path.join(save_dir, f"{tag}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self.log(f"[save] hf_dir = {hf_dir}")

    def _maybe_load_stage1_hf(self,hf_path):
        # save_root = os.path.join(getattr(self.args, "pretrain_checkpoints", "./pretrain_ckpts"), setting)

        # stage1_hf_dir = getattr(self.args, "stage1_hf_dir", None)
        stage1_hf_dir = hf_path
        if not stage1_hf_dir:
            return
        if not os.path.isdir(stage1_hf_dir):
            raise FileNotFoundError(f"stage1_hf_dir not found: {stage1_hf_dir}")
        if not self._is_two_stage_model():
            self.log(f"[warn] stage1_hf_dir is set but model is not two-stage. Ignore: {stage1_hf_dir}")
            return

        m = _unwrap(self.model)
        if hasattr(m, "stage"):
            m.stage = 2
        self.log(f"[load] stage1 hf from: {stage1_hf_dir}")

        if hasattr(m, "load_hf_dir"):
            m.load_hf_dir(stage1_hf_dir)
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            m.tokenizer = AutoTokenizer.from_pretrained(stage1_hf_dir, use_fast=False)
            m.llm = AutoModelForCausalLM.from_pretrained(stage1_hf_dir)

        emb = m.llm.get_input_embeddings()
        assert emb.num_embeddings >= len(m.tokenizer), "Embedding < vocab, please resize_token_embeddings in wrapper."

    def _maybe_load_stage1_wrapper(self, setting: str):
        paths = self._paths_for_setting(setting)
        wrapper_path = paths["stage1_wrapper"]
        if not os.path.isfile(wrapper_path):
            self.log(f"[warn] stage1 wrapper not found: {wrapper_path}")
            return

        m = _unwrap(self.model)
        self.log(f"[load] stage1 wrapper from: {wrapper_path}")
        if hasattr(m, "load_wrapper"):
            m.load_wrapper(wrapper_path, map_location=self.device)
        else:
            sd = torch.load(wrapper_path, map_location=self.device)
            m.load_state_dict(sd, strict=False)

    # -------------------------
    # freeze/unfreeze
    # -------------------------
    # def set_trainable_modules(self):
    #     if not self._is_two_stage_model():
    #         return
    #
    #     m = _unwrap(self.model)
    #     if not hasattr(m, "llm"):
    #         return
    #
    #     m.llm.requires_grad_(False)
    #
    #     tm = getattr(self.args, "trainable_modules", "")
    #     allow = [x.strip() for x in tm.split(",") if x.strip()]
    #     if not allow:
    #         allow = ["sensor_patch_proj", "channel_id", "patch_pos", "mask_embed", "recon_head", "cls_head", "pool_query", "pool_attn"]
    #
    #     for name in allow:
    #         if hasattr(m, name):
    #             getattr(m, name).requires_grad_(True)
    #         elif hasattr(m.llm, name):
    #             getattr(m.llm, name).requires_grad_(True)
    #         else:
    #             print(f"[warn] trainable module '{name}' not found in wrapper or llm")
    #
    #     n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    #     n_all = sum(p.numel() for p in self.model.parameters())
    #     self.log(f"[trainable] {n_train}/{n_all} = {100*n_train/n_all:.2f}%")

    def set_trainable_modules(self):
        if not self._is_two_stage_model():
            return

        m = _unwrap(self.model)

        # 1) 先全部冻结（包括 wrapper 所有外挂）
        for p in m.parameters():
            p.requires_grad = False

        # 2) LLM 永久冻结（如果你就是这个策略）
        if hasattr(m, "llm"):
            for p in m.llm.parameters():
                p.requires_grad = False

        # 3) 用“参数名前缀”来解冻：最稳，不怕模块被替换
        tm = str(getattr(self.args, "trainable_modules", "")).strip()
        allow = [x.strip() for x in tm.split(",") if x.strip()]
        if not allow:
            # 这里写你模型真实存在的外挂名字
            allow = [
                "sensor_patch_proj",
                "channel_id",
                "pos_mlp",
                "mask_embed",
                "resampler",
                "recon_decoder",
                "dec_attn",
                "recon_head",
                "pool_query",
                "pool_attn",
                "cls_head",
            ]
            allow = ["sensor_patch_proj", "channel_id", "patch_pos", "mask_embed", "recon_head", "cls_head", "pool_query", "pool_attn"]

        # 4) 解冻：只要参数名以这些前缀开头就放行
        hit = {k: 0 for k in allow}
        for n, p in m.named_parameters():
            for k in allow:
                if (n == k) or n.startswith(k + ".") or (("." + k + ".") in n):
                    p.requires_grad = True
                    hit[k] += p.numel()
                    break

        # 5) 强诊断
        self.log("[trainable-hit] " + ", ".join([f"{k}:{hit[k] / 1e6:.3f}M" for k in allow]))
        n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in m.parameters())
        self.log(f"[trainable] {n_train}/{n_all} = {100 * n_train / n_all:.4f}%")

        # 如果 0，直接报错并打印一些参数名方便你改 allow
        if n_train == 0:
            names = [n for n, _ in list(m.named_parameters())[:200]]
            self.log("[trainable][error] n_train == 0, allow=" + ",".join(allow))
            self.log("[trainable][error] first-200 param names:\n" + "\n".join(names))
            raise RuntimeError("No trainable params after set_trainable_modules(). Check allow names / nesting.")

    # -------------------------
    # forward wrappers
    # -------------------------
    def _forward_classify(self, batch_x, padding_mask):
        if self._is_two_stage_model():
            out = self.model(batch_x, padding_mask, mode="classify")
        else:
            out = self.model(batch_x, padding_mask, None, None)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() != 2:
            raise RuntimeError(f"classify output must be [B,num_class], got {tuple(out.shape)}")
        return out

    def _forward_pretrain_loss(self, batch_x, padding_mask):
        if not self._is_two_stage_model():
            raise RuntimeError("This model does not support pretrain stage.")
        out = self.model(batch_x, padding_mask, mode="pretrain")
        if not isinstance(out, (tuple, list)) or len(out) < 1:
            raise RuntimeError("pretrain forward must return (loss_mse, ...)")
        loss_mse = out[0]
        if not torch.is_tensor(loss_mse) or loss_mse.dim() != 0:
            raise RuntimeError(f"loss_mse must be scalar tensor, got {type(loss_mse)} shape={getattr(loss_mse,'shape',None)}")
        return out

    def log(self, msg: str):
        if self.logger is not None:
            self.logger.write(msg, also_print=True)
        else:
            print(msg)

    # -------------------------
    # masks / weights
    # -------------------------
    def _to_bool_mask(self, padding_mask):
        if padding_mask is None:
            return None
        padding_mask = padding_mask.to(self.device)
        if padding_mask.dtype != torch.bool:
            padding_mask = padding_mask > 0
        return padding_mask

    def _compute_class_weights(self, train_loader, num_class: int) -> torch.Tensor:
        counts = torch.zeros(num_class, dtype=torch.long)
        for _, label, _ in train_loader:
            y = label.long().view(-1)
            for k in y:
                kk = int(k)
                if 0 <= kk < num_class:
                    counts[kk] += 1
        w = 1.0 / counts.float().clamp_min(1.0)
        w = w / w.mean()
        return w

    # -------------------------
    # optimizer / criterion
    # -------------------------
    def _select_optimizer(self):
        lr = float(getattr(self.args, "learning_rate", 1e-3))
        wd = float(getattr(self.args, "weight_decay", 1e-2))

        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            nn_ = n.lower()
            # safe no_decay rule
            if n.endswith("bias") or ("norm" in nn_) or (".bn" in nn_) or ("layernorm" in nn_):
                no_decay.append(p)
            elif ("embedding" in nn_) or nn_.endswith("embed") or ("_embed" in nn_) or ("embed_" in nn_):
                no_decay.append(p)
            else:
                decay.append(p)

        groups = [
            {"params": decay, "lr": lr, "weight_decay": wd},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]
        opt = optim.RAdam(groups, lr=lr)
        # opt = optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))

        return opt

    def _select_criterion(self, train_loader=None):
        # num_class = int(getattr(self.args, "num_class", getattr(self.args, "num_labels", 12)))
        # if train_loader is not None and bool(getattr(self.args, "use_class_weight", True)):
        #     w = self._compute_class_weights(train_loader, num_class).to(self.device)
        #     return nn.CrossEntropyLoss(weight=w)
        return nn.CrossEntropyLoss()

    # -------------------------
    # scheduler builder
    # -------------------------
    def _build_scheduler(self, opt, steps_per_epoch: int):
        if not bool(getattr(self.args, "use_cosine", True)):
            return None

        min_lr = float(getattr(self.args, "min_lr", 1e-6))
        warmup_epochs = int(getattr(self.args, "warmup_epochs", 0))
        cosine_by_iter = bool(getattr(self.args, "cosine_by_iter", False))
        epochs = int(getattr(self.args, "train_epochs", 10))

        # --- option 1: epoch-step cosine (default, simplest) ---
        if not cosine_by_iter:
            cosine = CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=min_lr)

            if warmup_epochs <= 0:
                return cosine

            # warmup+cosine: use LambdaLR wrapper to scale lr first warmup_epochs
            def lr_lambda(e):
                if e < warmup_epochs:
                    return float(e + 1) / float(warmup_epochs)
                return 1.0

            warm = LambdaLR(opt, lr_lambda=lr_lambda)
            return {"warm": warm, "cosine": cosine}

        # --- option 2: iter-step cosine ---
        total_steps = max(1, epochs * steps_per_epoch)
        cosine = CosineAnnealingLR(opt, T_max=total_steps, eta_min=min_lr)
        return cosine

        # # --- option 2: iter-step warmup + cosine (PyTorch 2.0 OK) ---
        # total_steps = max(1, epochs * steps_per_epoch)
        # warmup_steps = max(0, warmup_epochs * steps_per_epoch)
        #
        # # warmup 后剩余给 cosine 的步数
        # cosine_steps = max(1, total_steps - warmup_steps)
        #
        # # cosine 只跑 “warmup 之后的部分”
        # cosine = CosineAnnealingLR(opt, T_max=cosine_steps, eta_min=min_lr)
        #
        # if warmup_steps <= 0:
        #     return cosine
        #
        # def warmup_lambda(step: int):
        #     # 线性 warmup: 从 0 -> 1
        #     return float(step + 1) / float(warmup_steps)
        #
        # warm = LambdaLR(opt, lr_lambda=warmup_lambda)
        #
        # # 在 warmup_steps 这个 milestone 切换到 cosine
        # sched = SequentialLR(opt, schedulers=[warm, cosine], milestones=[warmup_steps])
        # return sched
    def _log_lrs(self, opt, prefix: str):
        lrs = [pg["lr"] for pg in opt.param_groups]
        self.log(prefix + " " + ",".join([f"{x:.2e}" for x in lrs]))

    # ============================================================
    # Stage1: pretrain
    # ============================================================
    def pretrain_vali(self, loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_x, _, padding_mask in loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = self._to_bool_mask(padding_mask)
                out = self._forward_pretrain_loss(batch_x, padding_mask)
                loss_mse = out[0]

                losses.append(float(loss_mse.item()))
        self.model.train()
        return float(np.mean(losses)) if len(losses) else 0.0

    def pretrain_test(self, setting, test=0):
        if not self._is_two_stage_model():
            raise RuntimeError("Stage1 test called, but model is not two-stage.")
        if test:
            self._maybe_load_stage1_hf()

        _, test_loader = self._get_data(flag="TEST")
        test_mse = self.pretrain_vali(test_loader)

        # folder_path = os.path.join("./results_pretrain", setting)
        folder_path = os.path.join(self.run_root, "stage1", "results", setting)
        folder_path = to_secure_path(folder_path)

        os.makedirs(folder_path, exist_ok=True)

        self.log(f"[Stage1-Test] test_mse={test_mse:.6f}")
        self.log("---------------------------------------------------------------------------------------")

        with open(os.path.join(folder_path, "result_pretrain_mse.txt"), "a", encoding="utf-8") as f:
            f.write(setting + "\n")
            f.write(f"test_mse:{test_mse:.6f}\n\n")

        return test_mse

    def pretrain(self, setting):
        if not self._is_two_stage_model():
            raise RuntimeError("pretrain called, but model is not two-stage.")

        _, train_loader = self._get_data(flag="TRAIN")
        _, val_loader = self._get_data(flag="TEST")

        paths = self._paths_for_setting(setting)

        # stage1 logger 改成每 setting 一个文件（推荐）
        self.logger = TeeLogger(paths["stage1_log"])

        # stage1 ckpt_dir
        save_root = paths["stage1_ckpt_dir"]
        os.makedirs(save_root, exist_ok=True)

        # stage1 meta
        self._dump_meta(paths["meta_stage1"], stage="stage1", setting=setting, paths=paths)

        self.log(f"[Stage1-Pretrain] setting={setting}")

        self.args.stage = 1
        m = _unwrap(self.model)
        if hasattr(m, "stage"):
            m.stage = 1

        if bool(getattr(self.args, "freeze_llm", True)):
            self.set_trainable_modules()

        # ---- diagnostics: lock in hyperparams ----
        self.log(
            f"[HP] lr={float(self.args.learning_rate):.2e} "
            f"wd={float(self.args.weight_decay):.2e} "
            f"min_lr={float(self.args.min_lr):.2e} "
            f"warmup_epochs={int(self.args.warmup_epochs)} "
            f"cosine_by_iter={bool(self.args.cosine_by_iter)}"
        )

        opt = self._select_optimizer()
        scheduler = self._build_scheduler(opt, steps_per_epoch=len(train_loader))

        best_val = None
        report_trainable_params(self.model)
        self._log_lrs(opt, "[LR] init")


        for epoch in range(self.args.train_epochs):
            self.model.train()
            tr = []

            for batch_x, _, padding_mask in train_loader:
                opt.zero_grad()
                batch_x = batch_x.float().to(self.device)
                padding_mask = self._to_bool_mask(padding_mask)

                out = self._forward_pretrain_loss(batch_x, padding_mask)
                loss_mse = out[0]
                val_perplexity = out[-1]["perplexity"]
                loss_mse.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                opt.step()

                # iter-step scheduler
                if scheduler is not None and bool(getattr(self.args, "cosine_by_iter", False)):
                    scheduler.step()

                tr.append(float(loss_mse.item()))

            # epoch-step scheduler
            if scheduler is not None and not bool(getattr(self.args, "cosine_by_iter", False)):
                if isinstance(scheduler, dict):
                    # warmup then cosine
                    if epoch < int(getattr(self.args, "warmup_epochs", 0)):
                        scheduler["warm"].step()
                    else:
                        scheduler["cosine"].step()
                else:
                    scheduler.step()

            train_mse = float(np.mean(tr)) if len(tr) else 0.0
            val_mse = self.pretrain_vali(val_loader)
            self._log_lrs(opt, f"[LR] epoch={epoch + 1}")

            self.log(f"[Stage1-Pretrain] epoch={epoch + 1} train_mse={train_mse:.6f} val_mse={val_mse:.6f}")

            # ----------------------------
            # save best wrapper
            # ----------------------------
            # if best_val is None or val_mse < best_val:
            if best_val is None or (val_mse < best_val and val_perplexity > 10.0):
                best_val = val_mse

                m = _unwrap(self.model)
                wrapper_path = os.path.join(save_root, "best_wrapper.pth")
                wrapper_path = to_secure_path(wrapper_path)  # 处理

                if hasattr(m, "save_wrapper"):
                    m.save_wrapper(wrapper_path)
                else:
                    torch.save(m.state_dict(), wrapper_path)
                self.log(f"[save] wrapper = {wrapper_path}")

                # -------------------------------------------------------
                # 2. 【新增】单独保存 Codebook（用于论文可视化/分析）
                # -------------------------------------------------------
                # 假设你的 vqvae 结构是 self.model.vqvae.quantizer
                # 根据你的代码 vqvae.py，quantizer 可能是 List (分部位) 或 单个对象

                codebook_save_path = os.path.join(save_root, "best_codebook.pth")
                codebook_save_path = to_secure_path(codebook_save_path)  # 处理

                # 定义一个辅助函数，通吃各种量化器的实现方式
                def get_codebook_tensor(quantizer_module):
                    # 1. 优先检查 codebook (QuantizeEMAReset, QuantizeEMA, QuantizeReset)
                    if hasattr(quantizer_module, 'codebook'):
                        return quantizer_module.codebook.data.cpu()
                    # 2. 检查标准 embedding (Quantizer)
                    if hasattr(quantizer_module, 'embedding'):
                        return quantizer_module.embedding.weight.data.cpu()
                    # 3. 检查旧版实现 _w
                    if hasattr(quantizer_module, '_w'):
                        return quantizer_module._w.data.cpu()
                    return None

                try:
                    codebook_data = None

                    # 情况 A: 分部位 VQVAE (ModuleList: quantizers)
                    if hasattr(m, 'quantizers'):
                        self.log("[Checkpoint] Detected Multi-Part VQVAE (quantizers list).")
                        codebooks_dict = {}
                        for i, q in enumerate(m.quantizers):
                            cb = get_codebook_tensor(q)
                            if cb is not None:
                                codebooks_dict[f'part_{i}'] = cb
                            else:
                                self.log(f"[Warning] Could not find codebook weights for part {i}")

                        if codebooks_dict:
                            codebook_data = codebooks_dict

                    # 情况 B: 单一 VQVAE (Module: quantizer) -> 你目前是这个
                    elif hasattr(m, 'quantizer'):
                        self.log("[Checkpoint] Detected Vanilla VQVAE (single quantizer).")
                        q = m.quantizer
                        cb = get_codebook_tensor(q)

                        if cb is not None:
                            codebook_data = cb
                        else:
                            self.log(
                                "[Warning] Quantizer exists but codebook tensor not found (checked 'codebook', 'embedding', '_w')")

                    # 执行保存
                    if codebook_data is not None:
                        torch.save(codebook_data, codebook_save_path)
                        self.log(f"[Save] Codebook successfully saved to: {codebook_save_path}")
                        # 顺便打印一下形状，让你放心
                        if isinstance(codebook_data, dict):
                            shapes = {k: v.shape for k, v in codebook_data.items()}
                            self.log(f"[Debug] Codebook shapes: {shapes}")
                        else:
                            self.log(f"[Debug] Codebook shape: {codebook_data.shape}")
                    else:
                        self.log(
                            f"[Error] FAILED to extract codebook! Structure of m: {dir(m) if hasattr(m, 'vqvae') else 'No vqvae'}")

                except Exception as e:
                    self.log(f"[Error] Exception during codebook saving: {str(e)}")


                self.log(f"[save] wrapper = {wrapper_path}")
                self.log(f"[save] codebook extracted = {codebook_save_path}")

                status_path = os.path.join(paths["meta_dir"], "status_stage1.json")
                status_path = to_secure_path(status_path)  # 处理

                with open(status_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "best_val": float(best_val),
                        "best_epoch": int(epoch + 1),
                        "artifact_path": paths["stage1_wrapper"],
                        "wrapper_path": wrapper_path,
                        "codebook_save_path": codebook_save_path,
                        "artifact_exists": os.path.exists(paths["stage1_wrapper"]),
                        "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }, f, ensure_ascii=False, indent=2)


        return

    # ============================================================
    # Stage2: classify
    # ============================================================
    def vali_classify(self, loader, criterion, save_dir=None):

        total_loss, preds, trues = [], [], []
        self.model.eval()

        n_samples_total = 0
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        infer_start = time.time()

        with torch.no_grad():
            for batch_x, label, padding_mask in loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = self._to_bool_mask(padding_mask)
                label = label.to(self.device)

                outputs = self._forward_classify(batch_x, padding_mask)  # logits [B, C]
                target = label.long().view(-1)
                # label_test= label.long().squeeze(-1)
                # print("target:", target.shape)
                # print("label_test:", label_test.shape)
                # print(i)
                loss = criterion(outputs, target)
                # loss = criterion(outputs, label.long().squeeze(-1))

                total_loss.append(float(loss.item()))
                preds.append(outputs.detach())
                trues.append(label.detach())
                n_samples_total += batch_x.size(0)

        # -------- timing (IMPORTANT: sync at end) --------
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        total_time = time.time() - infer_start

        ms_per_sample = (total_time / max(n_samples_total, 1)) * 1000.0
        samples_per_sec = (n_samples_total / max(total_time, 1e-9))
        self.log(
            f"[Inference][Total] {ms_per_sample:.3f} ms/sample | "
            f"{samples_per_sec:.1f} samples/s "
            f"(total_samples={n_samples_total})"
        )

        # -------- aggregate --------
        total_loss = float(np.mean(total_loss)) if len(total_loss) else 0.0
        preds = torch.cat(preds, 0)  # logits [N, C]
        trues = torch.cat(trues, 0).flatten()  # [N]

        # numpy
        logits_np = preds.detach().cpu().numpy()
        trues_np = trues.detach().cpu().numpy()
        trues_np = np.squeeze(trues_np)  # 保险: (N,1) -> (N,)

        # pred / probs
        predictions = np.argmax(logits_np, axis=1)

        # stable softmax in numpy
        x = logits_np - np.max(logits_np, axis=1, keepdims=True)
        exp_x = np.exp(x)
        probs_np = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # -------- metrics --------
        metrics = {
            "acc": float(accuracy_score(trues_np, predictions)),
            "precision_macro": float(precision_score(trues_np, predictions, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(trues_np, predictions, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(trues_np, predictions, average="macro", zero_division=0)),
            "f1_micro": float(f1_score(trues_np, predictions, average="micro", zero_division=0)),
            "infer_total_time_s": float(total_time),
            "infer_ms_per_sample": float(ms_per_sample),
            "infer_samples_per_sec": float(samples_per_sec),
            "infer_total_samples": int(n_samples_total),
        }

        # per-class (optional)
        if bool(getattr(self.args, "report_per_class", False)):
            p_c = precision_score(trues_np, predictions, average=None, zero_division=0)
            r_c = recall_score(trues_np, predictions, average=None, zero_division=0)
            f_c = f1_score(trues_np, predictions, average=None, zero_division=0)
            for i, (p, r, f) in enumerate(zip(p_c, r_c, f_c)):
                metrics[f"precision_c{i}"] = float(p)
                metrics[f"recall_c{i}"] = float(r)
                metrics[f"f1_c{i}"] = float(f)

        # -------- save (ONLY when save_dir is provided) --------
        if save_dir is not None:
            save_dir = to_secure_path(save_dir)

            os.makedirs(save_dir, exist_ok=True)

            # save npy
            np.save(os.path.join(save_dir, "logits.npy"), logits_np)  # [N, C]
            np.save(os.path.join(save_dir, "probs.npy"), probs_np)  # [N, C]
            np.save(os.path.join(save_dir, "pred.npy"), predictions)  # [N]
            np.save(os.path.join(save_dir, "true.npy"), trues_np)  # [N]

            with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            # np.save(os.path.join(save_dir, "metrics.npy"), np.array([
            #     metrics["acc"],
            #     metrics["f1_macro"],
            #     metrics["recall_macro"],
            #     metrics["precision_macro"],
            #     metrics["f1_micro"],
            #     metrics["infer_total_time_s"],
            #     metrics["infer_ms_per_sample"],
            #     metrics["infer_samples_per_sec"],
            #     metrics["infer_total_samples"],
            # ], dtype=np.float32))

            # confusion
            conf = confusion_matrix(trues_np, predictions)
            np.save(os.path.join(save_dir, "confusion.npy"), conf)
        self.model.train()
        return total_loss, metrics

    def train(self, setting):

        # stage2 load stage1 automatically
        save_root, wrapper_path, hf_dir = self._autofind_stage1_paths(setting)



        if wrapper_path is not None:
            self.log(f"[auto-load] stage1 wrapper: {wrapper_path}")
            m = _unwrap(self.model)
            if hasattr(m, "stage"):
                m.stage = 2
            if hasattr(m, "load_wrapper"):
                m.load_wrapper(wrapper_path, map_location=self.device)
            else:
                sd = torch.load(wrapper_path, map_location=self.device)
                m.load_state_dict(sd, strict=False)

        elif hf_dir is not None:
            self.log(f"[auto-load] stage1 hf: {hf_dir}")
            self._maybe_load_stage1_hf(hf_dir)  # 你已有的函数会用 save_root
        else:
            self.log(f"[auto-load][warn] no stage1 found under: {save_root}")

        self.args.stage = 2
        # m = _unwrap(self.model)
        # if hasattr(m, "stage"):
        #     m.stage = 2

        _, train_loader = self._get_data(flag="TRAIN")
        _, val_loader = self._get_data(flag="TEST")
        _, test_loader = self._get_data(flag="TEST")

        paths = self._paths_for_setting(setting)

        # stage2 logger 每 setting 一个文件（推荐）
        self.logger = TeeLogger(paths["stage2_log"])

        # stage2 ckpt_dir
        path = paths["stage2_ckpt_dir"]
        path = to_secure_path(path)

        os.makedirs(path, exist_ok=True)

        # stage2 meta
        self._dump_meta(paths["meta_stage2"], stage="stage2", setting=setting, paths=paths)

        self.log(f"[Stage2-Train] setting={setting}")

        if bool(getattr(self.args, "freeze_llm", True)):
            self.set_trainable_modules()

        # ---- diagnostics ----
        self.log(f"[HP] lr={float(self.args.learning_rate):.2e} wd={float(self.args.weight_decay):.2e} min_lr={float(self.args.min_lr):.2e} warmup_epochs={int(self.args.warmup_epochs)} cosine_by_iter={bool(self.args.cosine_by_iter)} monitor={str(getattr(self.args,'monitor','acc'))}")

        opt = self._select_optimizer()
        scheduler = self._build_scheduler(opt, steps_per_epoch=len(train_loader))

        criterion = self._select_criterion(train_loader=train_loader)

        # early stop: default monitor acc (same as V1)
        monitor = str(getattr(self.args, "monitor", "loss")).lower()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        train_steps = len(train_loader)
        time_now = time.time()
        report_trainable_params(self.model)
        self._log_lrs(opt, "[LR] init")

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()
            train_loss = []

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                opt.zero_grad()
                batch_x = batch_x.float().to(self.device)
                padding_mask = self._to_bool_mask(padding_mask)
                label = label.to(self.device)

                outputs = self._forward_classify(batch_x, padding_mask)
                # print("outputs:", outputs.shape, "label:", label.shape)
                target = label.long().view(-1)
                # label_test= label.long().squeeze(-1)
                # print("target:", target.shape)
                # print("label_test:", label_test.shape)
                # print(i)
                loss = criterion(outputs, target)

                train_loss.append(float(loss.item()))

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                opt.step()

                # iter-step scheduler
                if scheduler is not None and bool(getattr(self.args, "cosine_by_iter", False)):
                    scheduler.step()

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / 100
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.log(f"\titers:{i + 1}, epoch:{epoch + 1} | loss:{loss.item():.6f} | "
                             f"speed:{speed:.4f}s/iter | left:{left_time:.1f}s")
                    # lr snapshot
                    self._log_lrs(opt, "[LR] iter")
                    time_now = time.time()

            # epoch-step scheduler
            if scheduler is not None and not bool(getattr(self.args, "cosine_by_iter", False)):
                if isinstance(scheduler, dict):
                    if epoch < int(getattr(self.args, "warmup_epochs", 0)):
                        scheduler["warm"].step()
                    else:
                        scheduler["cosine"].step()
                else:
                    scheduler.step()

            train_loss = float(np.mean(train_loss)) if len(train_loss) else 0.0
            # val_loss, val_acc = self.vali_classify(val_loader, criterion)
            # test_loss, test_acc = self.vali_classify(test_loader, criterion)
            #
            # self._log_lrs(opt, f"[LR] epoch={epoch+1}")

            val_loss, val_m = self.vali_classify(val_loader, criterion)
            test_loss, test_m = self.vali_classify(test_loader, criterion)

            self._log_lrs(opt, f"[LR] epoch={epoch+1}")

            self.log(
                f"[Stage2-Classify] Epoch:{epoch + 1} | Train:{train_loss:.4f} | "
                f"Val:{val_loss:.4f} acc:{val_m['acc']:.4f} f1m:{val_m['f1_macro']:.4f} recm:{val_m['recall_macro']:.4f} prem:{val_m['precision_macro']:.4f} | "
                f"Test:{test_loss:.4f} acc:{test_m['acc']:.4f} f1m:{test_m['f1_macro']:.4f} | "
                f"time:{time.time() - epoch_time:.1f}s"
            )

            #
            # # 默认训练中不评估 test（科研规范）
            # if bool(getattr(self.args, "eval_test_during_train", False)):
            #     test_loss, test_acc = self.vali_classify(test_loader, criterion)
            #     test_msg = f" | Test:{test_loss:.4f} Acc:{test_acc:.4f}"
            # else:
            #     test_msg = ""
            #
            # self.log(
            #     f"[Stage2-Classify] Epoch:{epoch + 1} | Train:{train_loss:.4f} | "
            #     f"Val:{val_loss:.4f} Acc:{val_acc:.4f}{test_msg} | "
            #     f"time:{time.time() - epoch_time:.1f}s"
            # )
            #
            # # self.log(f"[Stage2-Classify] Epoch:{epoch + 1} | Train:{train_loss:.4f} | "
            # #          f"Val:{val_loss:.4f} Acc:{val_acc:.4f} | Test:{test_loss:.4f} Acc:{test_acc:.4f} | "
            # #          f"time:{time.time() - epoch_time:.1f}s")

            # early stopping



            ####################################
            validate_vqvae(self.model, train_loader, device=self.args.device)
            ####################################


            if monitor == "loss":
                early_stopping(val_loss, self.model, path)
            else:
                early_stopping(-val_m["acc"], self.model, path)
            # ---- write stage2 status (best snapshot if updated) ----
            status_path = os.path.join(paths["meta_dir"], "status_stage2.json")
            status_path = to_secure_path(status_path)

            with open(status_path, "w", encoding="utf-8") as f:
                json.dump({
                    "monitor": str(monitor),
                    "last_epoch": int(epoch + 1),
                    "val_acc": float(val_m["acc"]),
                    "val_loss": float(val_loss),
                    "test_acc": float(test_m["acc"]),
                    "test_loss": float(test_loss),
                    "artifact_path": paths["stage2_ckpt"],
                    "artifact_exists": os.path.exists(paths["stage2_ckpt"]),
                    "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }, f, ensure_ascii=False, indent=2)

            if early_stopping.early_stop:
                self.log("Early stopping")
                break

        # best_model_path = os.path.join(path, "checkpoint.pth")
        # if os.path.exists(best_model_path):
        #     self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        best_model_path = os.path.join(path, "checkpoint.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        # # 最终只评估一次 test（用 val-best）
        # final_test_loss, final_test_acc = self.vali_classify(test_loader, criterion)
        # self.log(f"[Stage2-FinalTest] loss:{final_test_loss:.6f} acc:{final_test_acc:.6f}")
        return self.model

    def test_classify(self, setting, test=0):
        _, test_loader = self._get_data(flag="TEST")
        if test:
            self.log("loading model")
            # ckpt = os.path.join("./checkpoints", setting, "checkpoint.pth")
            ckpt = os.path.join(self.run_root, "stage2", "ckpts", setting, "checkpoint.pth")
            ckpt = to_secure_path(ckpt)

            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))

        _, train_loader = self._get_data(flag="TRAIN")
        criterion = self._select_criterion(train_loader=train_loader)
        # all_logits = []
        # all_trues = []
        #
        # n_samples_total = 0

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        # infer_start = time.time()
        # folder_path = os.path.join("./results", setting)
        folder_path = os.path.join(self.run_root, "stage2", "results", setting)
        folder_path = to_secure_path(folder_path)  # 关键：这里加上 \\?\

        os.makedirs(folder_path, exist_ok=True)

        test_loss, test_m = self.vali_classify(test_loader, criterion, save_dir=folder_path)



        self.log(f"[Stage2-Test] loss:{test_loss:.6f} acc:{test_m['acc']:.6f} f1_macro:{test_m['f1_macro']:.6f} recall_macro:{test_m['recall_macro']:.6f} precision_macro:{test_m['precision_macro']:.6f} infer_total_time:{test_m['infer_total_time_s']:.6f} infer_ms_per_sample:{test_m['infer_ms_per_sample']:.6f} infer_samples_per_sec:{test_m['infer_samples_per_sec']:.6f} infer_total_samples:{test_m['infer_total_samples']}")
        self.log("------------------------------------------------------------------------------")

        with open(os.path.join(folder_path, "result_classification.txt"), "a", encoding="utf-8") as f:
            f.write(setting + "\n")
            f.write(
                f"loss:{test_loss:.6f} "
                f"acc:{test_m['acc']:.6f} "
                f"f1_macro:{test_m['f1_macro']:.6f} "
                f"recall_macro:{test_m['recall_macro']:.6f} "
                f"precision_macro:{test_m['precision_macro']:.6f} "
                f"infer_total_time:{test_m['infer_total_time_s']:.6f} "
                f"infer_ms_per_sample:{test_m['infer_ms_per_sample']:.6f} "
                f"infer_samples_per_sec:{test_m['infer_samples_per_sec']:.6f} "
                f"infer_total_samples:{test_m['infer_total_samples']}\n\n"
            )

        return test_loss, test_m

    def test(self, setting, test=0):
        st = int(getattr(self.args, "stage", 2))
        if st == 1:
            return self.pretrain_test(setting, test=test)
        return self.test_classify(setting, test=test)


def get_configs():
    import random
    import numpy as np
    import argparse
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # datasets loader
    parser.add_argument('--datasets', type=str, required=False, default='ETTh1', help='dataset type')
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./datasets/ETT/', help='root path of the datasets file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='datasets file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output datasets', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='datasets loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input datasets')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=50, help='patch length')
    parser.add_argument('--stride', type=int, default=50, help='stride')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    # SensorLLMFuy
    parser.add_argument('--ts_backbone_yaml', type=str, default='ts_backbone.yaml', help='ts_backbone_yaml')
    parser.add_argument('--log_dir', type=str, default='./logs', help='log_dir')
    parser.add_argument('--dataset_key', type=str, default='mhealth', help='dataset_key')
    parser.add_argument('--stage', type=int, default=1, help='stage')
    parser.add_argument('--llama_name', type=str, default=r"D:\fuy\MyCode\SensorLLM\Llama-3.2-1B", help='stage')
    parser.add_argument('--two_stage', type=int, default=1, help='two_stage')
    parser.add_argument('--freeze_llm', type=int, default=0, help='freeze_llm')
    parser.add_argument('--trainable_modules', type=str,
                        default="sensor_patch_proj,channel_id,patch_pos,mask_embed,recon_head,cls_head,pool_query,pool_attn",
                        help='trainable_modules')
    parser.add_argument('--run_id', type=str, default="202501220_0956", help='trainable_modules')
    parser.add_argument('--test_subjects', type=str, default="202501220_0956", help='test_subjects')
    parser.add_argument('--pamap_variant', type=str, default="202501220_0956", help='pamap_variant')

    # datasets motion
    parser.add_argument('--test_users', type=str, default="19,20,21,22,23,24", help='test_users')
    parser.add_argument('--val_users', type=str, default="13,14,15,16,17,18", help='val_users')
    parser.add_argument('--motionsense_feature_set', type=str, default="A12", help='motionsense_feature_set')
    parser.add_argument('--motionsense_combine_grav_acc', type=int, default=0, help='motionsense_combine_grav_acc')
    parser.add_argument('--motionsense_norm', type=str, default='none', help='motionsense_norm')

    parser.add_argument('--hhar_tol', type=float, default=0.05, help='hhar_tol')
    parser.add_argument('--hhar_align_on', type=str, default="Arrival_Time", help='hhar_align_on')
    parser.add_argument('--hhar_use_cache', type=int, default=1, help='hhar_use_cache')
    parser.add_argument('--hhar_norm', type=str, default='none', help='hhar_norm')
    parser.add_argument('--wisdm_norm', type=str, default='none', help='wisdm_norm')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='val_ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test_ratio')

    args = parser.parse_args()
    return args
if __name__ == '__main__':
    configs = get_configs()
    LLAMA_NAME = r"D:\fuy\MyCode/Llama-3.2-1B"
    DATA_ROOT = r"D:\fuy\MyCode/SensorLLMLib/datasets/USC-HAD/USC-HAD"
    DATA_KEY = "uschad"
    DATA_NAME = "USCHAD"
    RUN_ID = "${GLOBAL_TIME_TAG}_uschad_sensorllm"
    LOG_DIR = "$GLOBAL_LOG_ROOT/uschad"
    model_name='SensorLLMFuy_test_withllm_mae'
    PRETRAIN_trainable_modules = "patch_embed,resampler,mae_decoder"
    TRAIN_trainable_modules = "patch_embed,resampler,llm_proj,cls_head"

    # Settings (重置变量)
    ALIGN_W_MAX = 200
    SEQ_LEN = 200
    PATCH_LEN = 100
    BATCH_SIZE = 16  # 注意这里变了
    LR = 0.001
    EPOCHS = 8
    TEST_SUBJECTS = "subject13,subject14"

    configs.task_name = 'classification'
    configs.test_subjects = TEST_SUBJECTS
    configs.is_training = 1
    configs.root_path = DATA_ROOT
    configs.model_id = DATA_NAME
    configs.run_id = "$RUN_ID"
    configs.datasets =DATA_NAME
    configs.model = model_name
    configs.data =DATA_NAME
    configs.dataset_key =DATA_KEY
    configs.seq_len =SEQ_LEN
    configs.patch_len =PATCH_LEN
    configs.stride =PATCH_LEN
    configs.stage = 1
    configs.batch_size =BATCH_SIZE
    configs.trainable_modules =PRETRAIN_trainable_modules
    configs.llama_name =LLAMA_NAME
    configs.learning_rate =LR
    configs.train_epochs=EPOCHS
    configs.num_workers = 0
    configs.hhar_tol = 0.05
    configs.hhar_align_on = 'Arrival_Time'
    configs.hhar_use_cache = 1
    configs.hhar_norm = 'none'
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}'.format(
        configs.task_name,
        configs.model_id,
        configs.model,
        configs.data,
        configs.features,
        configs.seq_len,
        configs.label_len,
        configs.pred_len,
        configs.d_model,
        configs.n_heads,
        configs.e_layers,
        configs.d_layers,
        configs.d_ff,
        configs.expand,
        configs.d_conv,
        configs.factor,
        configs.embed,
        configs.distil,
        configs.des)

    exp = Exp_VQ_VAE_Classification(configs)
    exp.pretrain(setting)
    c = 'end'
