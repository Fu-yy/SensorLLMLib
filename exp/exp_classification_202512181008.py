import json

import yaml

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb

warnings.filterwarnings('ignore')


def _unwrap(m):
    # 兼容 DataParallel/DistributedDataParallel
    return m.module if hasattr(m, "module") else m

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        # ① 先把 cfg 注入 args（super 之前）
        self._inject_dataset_cfg(args)
        super(Exp_Classification, self).__init__(args)

    def _inject_dataset_cfg(self, args):
        """
        Read dataset config ONCE in Exp, then store into args.
        This avoids repeated file reading in Dataset/Model.
        """
        # 你可以用 args.ts_backbone_yaml 指定文件
        ts_yaml = getattr(args, "ts_backbone_yaml", None)
        if ts_yaml is None:
            return  # 没配就跳过
        project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        CONFIG_PATH = project_path + os.sep+'configs'+os.sep + ts_yaml

        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"ts_backbone_yaml not found: {CONFIG_PATH}")

        dataset_key = str(getattr(args, "dataset_key", getattr(args, "data", "mhealth"))).lower()

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg_all = yaml.safe_load(f)

        if dataset_key not in cfg_all:
            raise KeyError(f"dataset '{dataset_key}' not found in {ts_yaml}")

        ds_cfg = cfg_all[dataset_key]

        # 写入 args，保证 Dataset/Model 都能直接用
        args.dataset_key = dataset_key
        args.ds_cfg = ds_cfg  # 直接挂 dict 也行（python进程内）
        # 常用字段：你至少保证这些存在/对齐
        if "channel_num" in ds_cfg:
            args.enc_in = int(ds_cfg["channel_num"])
        if "sample_rate" in ds_cfg:
            args.sample_rate = int(ds_cfg["sample_rate"])
        if "num_labels" in ds_cfg:
            args.num_class = int(ds_cfg["num_labels"])
        if "id2label" in ds_cfg:
            # 统一成 list 形式，避免 dict key 顺序坑
            id2label = ds_cfg["id2label"]
            # id2label 可能是 dict: {0:"xxx",1:"yyy"...}
            if isinstance(id2label, dict):
                max_id = max(int(k) for k in id2label.keys())
                args.class_names = [id2label[i] if i in id2label else id2label[str(i)]
                                    for i in range(max_id + 1)]
            else:
                args.class_names = list(id2label)
    # def _build_model(self):
    #     # model input depends on datasets
    #     train_data, train_loader = self._get_data(flag='TRAIN')
    #     test_data, test_loader = self._get_data(flag='TEST')
    #     self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
    #     self.args.pred_len = 0
    #     self.args.enc_in = train_data.feature_df.shape[1]
    #     self.args.num_class = len(train_data.class_names)
    #     # model init
    #     model = self.model_dict[self.args.model].Model(self.args).float()
    #     if self.args.use_multi_gpu and self.args.use_gpu:
    #         model = nn.DataParallel(model, device_ids=self.args.device_ids)
    #     return model
    def _build_model(self):
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')

        self.args.seq_len = max(getattr(train_data, "max_seq_len", self.args.seq_len),
                                getattr(test_data, "max_seq_len", self.args.seq_len))
        self.args.pred_len = 0

        # 优先用 cfg 注入的 enc_in / num_class
        if hasattr(self.args, "ds_cfg") and isinstance(self.args.ds_cfg, dict):
            self.args.enc_in = int(self.args.ds_cfg.get("channel_num", self.args.enc_in))
            self.args.num_class = int(self.args.ds_cfg.get("num_labels", getattr(self.args, "num_class", 12)))
        else:
            self.args.enc_in = train_data.feature_df.shape[1]
            self.args.num_class = len(train_data.class_names)

        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def save_hf_bundle(self, save_dir: str, tag: str = "best"):
        os.makedirs(save_dir, exist_ok=True)
        m = _unwrap(self.model)  # 你的 wrapper Model(nn.Module)

        # 1) HF 模型 + tokenizer 存到同一个目录下（最省事）
        hf_dir = os.path.join(save_dir, f"{tag}_hf")
        os.makedirs(hf_dir, exist_ok=True)

        # 关键：保存的是 m.llm（AutoModelForCausalLM），不是外层 wrapper
        m.llm.save_pretrained(hf_dir)
        m.tokenizer.save_pretrained(hf_dir)

        # 2) meta（非参数配置）
        meta = {
            "dataset_key": getattr(self.args, "dataset_key", getattr(self.args, "data", None)),
            "ds_cfg": getattr(self.args, "ds_cfg", None),
            "C": getattr(m, "C", getattr(self.args, "enc_in", None)),
            "ts_tokens": getattr(m, "ts_tokens", None),
            "pt_objective": getattr(self.args, "pt_objective", None),
        }
        with open(os.path.join(save_dir, f"{tag}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[save] hf_dir={hf_dir}")
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                if self.args.model == 'SensorLLMFuy':
                    outputs, mask, x_masked, meta = self.model(batch_x, padding_mask, None, None)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach()
                loss = criterion(pred, label.long().squeeze())
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):

                if epoch == 0 and i == 0:
                    print("batch_x:", batch_x.shape)
                    print("label:", label.shape, label.min().item(), label.max().item())
                    print("padding_mask:", padding_mask.shape, padding_mask.sum() / padding_mask.numel())

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # outputs = self.model(batch_x, padding_mask, None, None)
                if self.args.model == 'SensorLLMFuy':
                    outputs, mask, x_masked, meta = self.model(batch_x, padding_mask, None, None)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # outputs = self.model(batch_x, padding_mask, None, None)
                if self.args.model == 'SensorLLMFuy':
                    outputs, mask, x_masked, meta = self.model(batch_x, padding_mask, None, None)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return
