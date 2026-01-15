# exp/exp_sensorllm_unified.py
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

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, cal_accuracy

warnings.filterwarnings("ignore")

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

# =========================
# Utils
# =========================
def _unwrap(m):
    return m.module if hasattr(m, "module") else m


class TeeLogger:
    def __init__(self, log_path: str, flush: bool = True):
        self.log_path = log_path
        self.flush = flush
        d = os.path.dirname(log_path)
        if d:
            os.makedirs(d, exist_ok=True)

    def write(self, msg: str, also_print: bool = True):
        if msg is None:
            return
        msg = str(msg).rstrip("\n")

        if also_print:
            print(msg, flush=self.flush)

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
            if self.flush:
                f.flush()

# =========================
# Exp
# =========================
class Exp_Classification(Exp_Basic):
    """
    Unified classification exp:
      - For SensorLLM-like two-stage models:
          stage1: pretrain(setting) -> MSE reconstruction
          stage2: train(setting)    -> CE classification
      - For other models:
          only stage2 (classification)

    Other models' forward signature:
      outputs = model(batch_x, padding_mask, None, None) -> logits [B,num_class]
    """

    def __init__(self, args):
        # 0) inject ds cfg once
        self._inject_dataset_cfg(args)
        super().__init__(args)

        # -------------------------
        # logger init (train+pretrain)
        # -------------------------
        # -------------------------
        # logger init (ONE run)
        # -------------------------
        log_root = getattr(self.args, "log_dir", "./logs")

        # 如果用户显式传 run_id，则两次启动(stage1/stage2)都会落到同一个目录
        # 否则默认用当前时间戳（单次运行也不会冲突）
        run_id = getattr(self.args, "run_id", None)
        if not run_id:
            run_id = time.strftime("%Y%m%d_%H%M%S")
            setattr(self.args, "run_id", run_id)

        self.log_dir = os.path.join(
            log_root,
            getattr(self.args, "model", "model"),
            str(run_id)
        )
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = None

        # 2) sanity: if stage=2 and wants load stage1 hf
        if int(getattr(self.args, "stage", 2)) == 2:
            self._maybe_load_stage1_hf()

        # 3) freeze control (only meaningful for LLM-like wrapper)
        if bool(getattr(self.args, "freeze_llm", True)):
            self.set_trainable_modules()

    def _compute_class_weights(self, train_loader, num_class: int):
        counts = torch.zeros(num_class, dtype=torch.long)
        for _, label, _ in train_loader:
            y = label.long().view(-1)
            for k in y:
                if 0 <= int(k) < num_class:
                    counts[int(k)] += 1
        # 反比权重 + 归一化
        w = 1.0 / (counts.float().clamp_min(1.0))
        w = w / w.mean()
        return w

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

        dataset_key = str(getattr(args, "dataset_key", getattr(args, "data", "mhealth"))).lower()
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

    # -------------------------
    # two-stage detection
    # -------------------------
    def _is_two_stage_model(self) -> bool:
        """
        Preferred: explicitly set args.two_stage=1 for SensorLLM-like model.
        Fallback: infer by model name contains 'sensorllm' or has llm attr.
        """
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
        val_data, _ = self._get_data(flag="VAL")
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

        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # must support TRAIN/VAL/TEST
        return data_provider(self.args, flag)

    # -------------------------
    # optional: hf bundle save/load (two-stage only)
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

    def _maybe_load_stage1_hf(self):
        """
        If args.stage1_hf_dir is set, load pretrained hf into wrapper.
        Only for two-stage model.
        """
        stage1_hf_dir = getattr(self.args, "stage1_hf_dir", None)
        if not stage1_hf_dir:
            return
        if not os.path.isdir(stage1_hf_dir):
            raise FileNotFoundError(f"stage1_hf_dir not found: {stage1_hf_dir}")
        if not self._is_two_stage_model():
            # other models shouldn't load hf
            self.log(f"[warn] stage1_hf_dir is set but model is not two-stage. Ignore: {stage1_hf_dir}")
            return

        m = _unwrap(self.model)
        self.log(f"[load] stage1 hf from: {stage1_hf_dir}")

        if hasattr(m, "load_hf_dir"):
            m.load_hf_dir(stage1_hf_dir)
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            m.tokenizer = AutoTokenizer.from_pretrained(stage1_hf_dir, use_fast=False)
            m.llm = AutoModelForCausalLM.from_pretrained(stage1_hf_dir)

        # safety: embedding size >= vocab
        if hasattr(m, "llm") and hasattr(m, "tokenizer"):
            emb = m.llm.get_input_embeddings()
            assert emb.num_embeddings >= len(m.tokenizer), \
                "Embedding < vocab, please resize_token_embeddings in wrapper."

    # -------------------------
    # freeze/unfreeze
    # -------------------------
    def set_trainable_modules(self):
        """
        For two-stage models (LLM wrapper): freeze llm and open only small modules.
        For other models: do nothing (train all).
        """
        if not self._is_two_stage_model():
            return

        m = _unwrap(self.model)
        if not hasattr(m, "llm"):
            return

        # 1) freeze llm
        m.llm.requires_grad_(False)

        # 2) open whitelist modules
        tm = getattr(self.args, "trainable_modules", "")
        allow = [x.strip() for x in tm.split(",") if x.strip()]
        if not allow:
            allow = ["sensor_patch_proj", "channel_id","patch_pos","mask_embed", "recon_head", "cls_head"]

        for name in allow:
            if hasattr(m, name):
                getattr(m, name).requires_grad_(True)
            elif hasattr(m.llm, name):
                getattr(m.llm, name).requires_grad_(True)
            else:
                print(f"[warn] trainable module '{name}' not found in wrapper or llm")

        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.model.parameters())
        self.log(f"[trainable] {n_train}/{n_all} = {100*n_train/n_all:.2f}%")

    # -------------------------
    # unified forward
    # -------------------------
    def _forward_classify(self, batch_x, padding_mask):
        """
        Return logits [B,num_class]
        - two-stage: model(x, mask, mode="classify")
        - other:     model(x, mask, None, None)
        """
        if self._is_two_stage_model():
            out = self.model(batch_x, padding_mask, mode="classify")
        else:
            out = self.model(batch_x, padding_mask, None, None)

        # allow wrapper returns tuple
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() != 2:
            raise RuntimeError(f"classify output must be [B,num_class], got {tuple(out.shape)}")
        return out

    def _forward_pretrain_loss(self, batch_x, padding_mask):
        """
        Return loss_mse (scalar tensor)
        two-stage only.
        """
        if not self._is_two_stage_model():
            raise RuntimeError("This model does not support pretrain stage.")
        out = self.model(batch_x, padding_mask, mode="pretrain")
        if not isinstance(out, (tuple, list)) or len(out) < 1:
            raise RuntimeError("pretrain forward must return (loss_mse, ...)")
        loss_mse = out[0]
        if not torch.is_tensor(loss_mse) or loss_mse.dim() != 0:
            raise RuntimeError(f"loss_mse must be scalar tensor, got {type(loss_mse)} shape={getattr(loss_mse,'shape',None)}")
        return loss_mse

    def log(self, msg: str):
        # 如果你用 DataParallel/单卡都没问题
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.write(msg, also_print=True)
        else:
            print(msg)
    # -------------------------
    # optimizer / criterion
    # -------------------------
    def _select_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return optim.RAdam(params, lr=self.args.learning_rate)

    def _select_criterion(self,weight=None):
        return nn.CrossEntropyLoss(weight)

    # ============================================================
    # Stage1: Pretrain
    # ============================================================
    def pretrain_vali(self, loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_x, _, padding_mask in loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                loss_mse = self._forward_pretrain_loss(batch_x, padding_mask)
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

        folder_path = os.path.join("./results_pretrain", setting)
        os.makedirs(folder_path, exist_ok=True)

        # print(f"[Stage1-Test] test_mse={test_mse:.6f}")
        self.log(f"[Stage1-Test] test_mse={test_mse:.6f}")
        self.log(f"---------------------------------------------------------------------------------------")

        with open(os.path.join(folder_path, "result_pretrain_mse.txt"), "a", encoding="utf-8") as f:
            f.write(setting + "\n")
            f.write(f"test_mse:{test_mse:.6f}\n\n")

        return test_mse

    def pretrain(self, setting):
        if not self._is_two_stage_model():
            raise RuntimeError("pretrain called, but model is not two-stage.")

        train_data, train_loader = self._get_data(flag="TRAIN")
        val_data, val_loader = self._get_data(flag="VAL")

        save_root = os.path.join(getattr(self.args, "pretrain_checkpoints", "./pretrain_ckpts"), setting)
        os.makedirs(save_root, exist_ok=True)

        # switch log file
        log_root = getattr(self.args, "log_dir", "./logs")
        os.makedirs(log_root, exist_ok=True)
        # ---- stage1 logger ----
        # self.logger = TeeLogger(os.path.join(log_root, f"{setting}_stage1.log"))
        self.logger = TeeLogger(os.path.join(self.log_dir, "stage1.log"))
        self.log(f"[Stage1-Pretrain] setting={setting}")

        # mark stage
        self.args.stage = 1
        m = _unwrap(self.model)
        if hasattr(m, "stage"):
            m.stage = 1

        if bool(getattr(self.args, "freeze_llm", True)):
            self.set_trainable_modules()

        opt = self._select_optimizer()

        best_val = None

        report_trainable_params(self.model)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            tr = []

            for batch_x, _, padding_mask in train_loader:
                opt.zero_grad()
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                loss_mse = self._forward_pretrain_loss(batch_x, padding_mask)
                loss_mse.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                opt.step()
                tr.append(float(loss_mse.item()))

            train_mse = float(np.mean(tr)) if len(tr) else 0.0
            val_mse = self.pretrain_vali(val_loader)
            # print(f"[Stage1-Pretrain] epoch={epoch+1} train_mse={train_mse:.6f} val_mse={val_mse:.6f}")
            self.log(f"[Stage1-Pretrain] epoch={epoch+1} train_mse={train_mse:.6f} val_mse={val_mse:.6f}")

            if best_val is None or val_mse < best_val:
                best_val = val_mse
                self.save_hf_bundle(save_root, tag="best")

        return

    # ============================================================
    # Stage2: Classification
    # ============================================================
    def vali_classify(self, loader, criterion):
        total_loss = []
        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, label, padding_mask in loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self._forward_classify(batch_x, padding_mask)  # [B,num_class]
                loss = criterion(outputs, label.long().squeeze(-1))
                total_loss.append(float(loss.item()))

                preds.append(outputs.detach())
                trues.append(label.detach())

        total_loss = float(np.mean(total_loss)) if len(total_loss) else 0.0
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0).flatten()

        probs = torch.softmax(preds, dim=-1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues_np = trues.cpu().numpy()
        acc = cal_accuracy(predictions, trues_np)

        self.model.train()
        return total_loss, acc

    def train(self, setting):
        # mark stage
        self.args.stage = 2
        m = _unwrap(self.model)
        if hasattr(m, "stage"):
            m.stage = 2

        train_data, train_loader = self._get_data(flag="TRAIN")
        val_data, val_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        # switch log file
        log_root = getattr(self.args, "log_dir", "./logs")
        os.makedirs(log_root, exist_ok=True)
        # ---- stage2 logger ----
        # self.logger = TeeLogger(os.path.join(log_root, f"{setting}_stage2.log"))
        self.logger = TeeLogger(os.path.join(self.log_dir, "stage2.log"))
        self.log(f"[Stage2-Train] setting={setting}")

        # freeze control
        if bool(getattr(self.args, "freeze_llm", True)):
            self.set_trainable_modules()

        opt = self._select_optimizer()
        if self.args.model == "SensorLLMFuy":
            # num_class = self.args.enc_in
            # class_w = self._compute_class_weights(train_loader, num_class).to(self.device)
            class_w = None
        else:
            class_w =None
        criterion = self._select_criterion(class_w)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        train_steps = len(train_loader)
        time_now = time.time()
        report_trainable_params(self.model)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()
            train_loss = []

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                opt.zero_grad()
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self._forward_classify(batch_x, padding_mask)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(float(loss.item()))

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                opt.step()

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / 100
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print(f"\titers:{i+1}, epoch:{epoch+1} | loss:{loss.item():.6f} | "
                    #       f"speed:{speed:.4f}s/iter | left:{left_time:.1f}s")
                    self.log(f"\titers:{i + 1}, epoch:{epoch + 1} | loss:{loss.item():.6f} | "
                             f"speed:{speed:.4f}s/iter | left:{left_time:.1f}s")

                    time_now = time.time()

            train_loss = float(np.mean(train_loss)) if len(train_loss) else 0.0
            val_loss, val_acc = self.vali_classify(val_loader, criterion)
            test_loss, test_acc = self.vali_classify(test_loader, criterion)

            # print(f"[Stage2-Classify] Epoch:{epoch+1} | Train:{train_loss:.4f} | "
            #       f"Val:{val_loss:.4f} Acc:{val_acc:.4f} | Test:{test_loss:.4f} Acc:{test_acc:.4f} | "
            #       f"time:{time.time()-epoch_time:.1f}s")
            self.log(f"[Stage2-Classify] Epoch:{epoch + 1} | Train:{train_loss:.4f} | "
                     f"Val:{val_loss:.4f} Acc:{val_acc:.4f} | Test:{test_loss:.4f} Acc:{test_acc:.4f} | "
                     f"time:{time.time() - epoch_time:.1f}s")

            # early stopping monitors -val_acc (same as your old logic)
            early_stopping(-val_acc, self.model, path)
            if early_stopping.early_stop:
                self.log("Early stopping")
                break

        best_model_path = os.path.join(path, "checkpoint.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    def test_classify(self, setting, test=0):
        _, test_loader = self._get_data(flag="TEST")
        if test:
            self.log("loading model")
            ckpt = os.path.join("./checkpoints", setting, "checkpoint.pth")
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))

        criterion = self._select_criterion()
        test_loss, test_acc = self.vali_classify(test_loader, criterion)

        folder_path = os.path.join("./results", setting)
        os.makedirs(folder_path, exist_ok=True)

        # print(f"[Stage2-Test] loss:{test_loss:.6f} acc:{test_acc:.6f}")
        self.log(f"[Stage2-Test] loss:{test_loss:.6f} acc:{test_acc:.6f}")
        self.log(f"------------------------------------------------------------------------------")

        with open(os.path.join(folder_path, "result_classification.txt"), "a", encoding="utf-8") as f:
            f.write(setting + "\n")
            f.write(f"loss:{test_loss:.6f} acc:{test_acc:.6f}\n\n")

        return test_loss, test_acc

    # unified test entry
    def test(self, setting, test=0):
        st = int(getattr(self.args, "stage", 2))
        if st == 1:
            return self.pretrain_test(setting, test=test)
        return self.test_classify(setting, test=test)
