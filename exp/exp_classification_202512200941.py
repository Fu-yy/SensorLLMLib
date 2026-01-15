# exp/exp_sensorllm_two_stage.py
import os
import time
import json
import yaml
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy

warnings.filterwarnings("ignore")

import datetime

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

def _unwrap(m):
    return m.module if hasattr(m, "module") else m

def _count_labels(dl, max_batches=200):
    from collections import Counter
    c = Counter()
    n = 0
    for i, (_, y, _) in enumerate(dl):
        y = y.view(-1).cpu().numpy().tolist()
        c.update(y)
        n += len(y)
        if i+1 >= max_batches:
            break
    return n, c
class Exp_Classification(Exp_Basic):
    """
    Two-stage:
      - stage1: exp.pretrain(setting)  -> MSE
      - stage2: exp.train(setting)     -> CE
    Also supports: exp.test(setting)   -> dispatch by args.stage
    """

    def __init__(self, args):
        self._inject_dataset_cfg(args)
        super().__init__(args)

        # -------------------------
        # logger init (train+pretrain)
        # -------------------------
        # -------------------------
        # logger init (ONE run)
        # -------------------------
        log_root = getattr(self.args, "log_dir", "./logs")
        run_tag = time.strftime("%Y%m%d_%H%M%S")

        self.log_dir = os.path.join(
            log_root,
            getattr(self.args, "model", "model"),
            run_tag
        )
        os.makedirs(self.log_dir, exist_ok=True)

        # 默认还没进 stage
        self.logger = None

        # stage2 can optionally load stage1 hf
        if int(getattr(self.args, "stage", 2)) == 2:
            self._maybe_load_stage1_hf()

        # freeze/unfreeze
        if bool(getattr(self.args, "freeze_llm", True)):
            self.set_trainable_modules()

    # ------------------------------------------------------------
    # inject yaml cfg ONCE
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # build model
    # ------------------------------------------------------------
    def _build_model(self):
        train_data, _ = self._get_data(flag="TRAIN")
        val_data, _ = self._get_data(flag="VAL")
        test_data, _ = self._get_data(flag="TEST")

        self.args.seq_len = max(
            getattr(train_data, "max_seq_len", self.args.seq_len),
            getattr(val_data, "max_seq_len", self.args.seq_len),
            getattr(test_data, "max_seq_len", self.args.seq_len),
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
        # 你选择的是 B，所以 data_provider 需要支持 TRAIN/VAL/TEST
        return data_provider(self.args, flag)

    # ------------------------------------------------------------
    # HF save bundle (llm+tokenizer+meta)
    # ------------------------------------------------------------
    def save_hf_bundle(self, save_dir: str, tag: str = "best"):
        os.makedirs(save_dir, exist_ok=True)
        m = _unwrap(self.model)

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

    # ------------------------------------------------------------
    # stage2: load stage1 hf (optional)
    # ------------------------------------------------------------
    def _maybe_load_stage1_hf(self):
        stage1_hf_dir = getattr(self.args, "stage1_hf_dir", None)
        if not stage1_hf_dir:
            return
        if not os.path.isdir(stage1_hf_dir):
            raise FileNotFoundError(f"stage1_hf_dir not found: {stage1_hf_dir}")

        m = _unwrap(self.model)
        self.log(f"[load] stage1 hf from: {stage1_hf_dir}")

        if hasattr(m, "load_hf_dir"):
            m.load_hf_dir(stage1_hf_dir)
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            m.tokenizer = AutoTokenizer.from_pretrained(stage1_hf_dir, use_fast=False)
            m.llm = AutoModelForCausalLM.from_pretrained(stage1_hf_dir)

        assert m.llm.get_input_embeddings().num_embeddings >= len(m.tokenizer), \
            "Embedding < vocab, please resize."

    # ------------------------------------------------------------
    # freeze backbone; allow whitelist modules
    # ------------------------------------------------------------
    def set_trainable_modules(self):
        m = _unwrap(self.model)
        if not hasattr(m, "llm"):
            return

        # 1) freeze all LLM
        m.llm.requires_grad_(False)

        # 2) allow external modules (NOT inside llm unless你真的放进去了)
        tm = getattr(self.args, "trainable_modules", "")
        allow = [x.strip() for x in tm.split(",") if x.strip()]
        if not allow:
            allow = ["sensor_proj", "channel_id", "recon_head", "cls_head"]

        # 优先在 wrapper 上找（推荐）
        for name in allow:
            if hasattr(m, name):
                getattr(m, name).requires_grad_(True)
            elif hasattr(m.llm, name):
                # 如果你把模块挂在 llm 里，这里也能放开
                getattr(m.llm, name).requires_grad_(True)
            else:
                print(f"[warn] trainable module '{name}' not found in wrapper or llm")

        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.model.parameters())
        self.log(f"[trainable] {n_train}/{n_all} = {100*n_train/n_all:.2f}%")


    def log(self, msg: str):
        # 如果你用 DataParallel/单卡都没问题
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.write(msg, also_print=True)
        else:
            print(msg)

    # ------------------------------------------------------------
    # optimizer
    # ------------------------------------------------------------
    def _select_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return optim.RAdam(params, lr=self.args.learning_rate)

    # ============================================================
    # Stage 1: Pretrain (MSE)
    # ============================================================
    def pretrain_vali(self, loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_x, label, padding_mask in loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                loss_mse, _, _, _ = self.model(batch_x, padding_mask, mode="pretrain")
                losses.append(float(loss_mse.item()))
        self.model.train()
        return float(np.mean(losses)) if len(losses) else 0.0

    def pretrain_test(self, setting, test=0):
        """
        Stage1 test: report test_mse and save to results file.
        If test=1, you may choose to load stage1_hf_dir (best_hf) before testing.
        """
        # load best hf if provided
        if test:
            # 你可以传 args.stage1_hf_dir 指向 best_hf
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
        train_data, train_loader = self._get_data(flag="TRAIN")
        val_data, val_loader = self._get_data(flag="VAL")

        save_root = os.path.join(getattr(self.args, "pretrain_checkpoints", "./pretrain_ckpts"), setting)
        os.makedirs(save_root, exist_ok=True)


        # --- switch log file for this setting ---
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
        for epoch in range(self.args.train_epochs):
            self.model.train()
            tr = []

            for batch_x, label, padding_mask in train_loader:
                opt.zero_grad()
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                loss_mse, _, _, _ = self.model(batch_x, padding_mask, mode="pretrain")
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
    # Stage 2: Classification (CE)
    # ============================================================
    def _select_criterion(self):
        return nn.CrossEntropyLoss()

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

                outputs = self.model(batch_x, padding_mask, mode="classify")  # [B,num_class]
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
        self.args.stage = 2
        m = _unwrap(self.model)
        if hasattr(m, "stage"):
            m.stage = 2

        train_data, train_loader = self._get_data(flag="TRAIN")
        val_data, val_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")

        # n_tr, c_tr = _count_labels(train_loader)
        # n_va, c_va = _count_labels(val_loader)
        # n_te, c_te = _count_labels(test_loader)
        # self.log(f"[label_dist] train n={n_tr} {dict(c_tr)}")
        # self.log(f"[label_dist] val   n={n_va} {dict(c_va)}")
        # self.log(f"[label_dist] test  n={n_te} {dict(c_te)}")

        path = os.path.join(self.args.checkpoints, setting)



        os.makedirs(path, exist_ok=True)


        # --- switch log file for this setting ---
        log_root = getattr(self.args, "log_dir", "./logs")
        os.makedirs(log_root, exist_ok=True)
        # ---- stage2 logger ----
        # self.logger = TeeLogger(os.path.join(log_root, f"{setting}_stage2.log"))
        self.logger = TeeLogger(os.path.join(self.log_dir, "stage2.log"))
        self.log(f"[Stage2-Train] setting={setting}")

        if bool(getattr(self.args, "freeze_llm", True)):
            self.set_trainable_modules()

        opt = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        train_steps = len(train_loader)
        time_now = time.time()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()
            train_loss = []

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                opt.zero_grad()
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, mode="classify")
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
                    self.log(f"\titers:{i+1}, epoch:{epoch+1} | loss:{loss.item():.6f} | "
                             f"speed:{speed:.4f}s/iter | left:{left_time:.1f}s")

                    time_now = time.time()

            train_loss = float(np.mean(train_loss)) if len(train_loss) else 0.0
            val_loss, val_acc = self.vali_classify(val_loader, criterion)
            test_loss, test_acc = self.vali_classify(test_loader, criterion)

            # print(f"[Stage2-Classify] Epoch:{epoch+1} | Train:{train_loss:.4f} | "
            #       f"Val:{val_loss:.4f} Acc:{val_acc:.4f} | Test:{test_loss:.4f} Acc:{test_acc:.4f} | "
            #       f"time:{time.time()-epoch_time:.1f}s")
            self.log(f"[Stage2-Classify] Epoch:{epoch+1} | Train:{train_loss:.4f} | "
                     f"Val:{val_loss:.4f} Acc:{val_acc:.4f} | Test:{test_loss:.4f} Acc:{test_acc:.4f} | "
                     f"time:{time.time()-epoch_time:.1f}s")

            early_stopping(-val_acc, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(path, "checkpoint.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    # ------------------------------------------------------------
    # Stage2 final test (classification) - standalone like TSLib
    # ------------------------------------------------------------
    def test_classify(self, setting, test=0):
        _, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading model")
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

    # ------------------------------------------------------------
    # Unified test entry for run.py (dispatch by args.stage)
    # ------------------------------------------------------------
    def test(self, setting, test=0):
        st = int(getattr(self.args, "stage", 2))
        if st == 1:
            return self.pretrain_test(setting, test=test)
        else:
            return self.test_classify(setting, test=test)
