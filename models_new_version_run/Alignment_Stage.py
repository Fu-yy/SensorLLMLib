import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict, Any, Optional

from models_new_version_run.VQ_VAE import IMU_VQ_Model
from utils.label_utils import get_label_names_from_cfg

try:
    import yaml
except Exception:
    yaml = None


# --task_name=alignment
# --is_training=1
# --root_path="D:/fuy/MyCode/SensorLLMLib/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
# --root_path="D:/fuy/MyCode/SensorLLMLib/datasets/MHEALTHDATASET"
# --model_id=MHealth
# --run_id="alignment_weight"
# --datasets=MHealth
# --model="Alignment_Stage"
# --data=MHealth
# --dataset_key=mealth
# --seq_len=100
# --patch_len=50
# --stride=50
# --stage=1
# --batch_size=16
# --llama_name="D:\fuy\MyCode\Llama-3.2-1B"
# --learning_rate=0.001
# --train_epochs=10
# --num_workers=0
# --vqvae_path=qua_recon_path
# --test_subjects="subject1,subject3,subject6"
# models_new_version_run/Alignment_Stage.py
import os
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml
except Exception:
    yaml = None

from models_new_version_run.VQ_VAE import IMU_VQ_Model



import os
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml
except Exception:
    yaml = None

from models_new_version_run.VQ_VAE import IMU_VQ_Model
from models_new_version_run.PrimitiveLlmTeacher import (
    PrimitiveLLMTeacherCore,
    load_codebook_from_path_or_vq,
    pad_to_multiple,
)


class AlignmentModel(nn.Module):
    """
    Primitive-language teacher alignment model.

    Main training objective:
        loss = loss_primitive + beta * loss_activity

    Frozen:
        - VQ-VAE
        - LLM backbone

    Trainable:
        - primitive projector
        - mask embedding
        - primitive position embedding
        - query embedding
        - primitive recovery head
        - activity classification head
    """

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.stage = int(getattr(args, "stage", 1))
        self.device = args.device

        print(f"[AlignmentModel] Init in Stage: {self.stage}")

        # ============================================================
        # Dataset config
        # ============================================================
        self.dataset_key = str(
            getattr(args, "dataset_key", getattr(args, "data", "mhealth"))
        ).lower()

        self.ds_cfg: Dict[str, Any] = {}

        if hasattr(args, "ds_cfg") and isinstance(args.ds_cfg, dict):
            self.ds_cfg = args.ds_cfg
        else:
            ts_yaml = getattr(args, "ts_backbone_yaml", None)

            if ts_yaml is not None:
                if yaml is None:
                    raise ImportError("pyyaml is not installed but ts_backbone_yaml is set.")

                if os.path.exists(ts_yaml):
                    config_path = ts_yaml
                else:
                    project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
                    config_path = os.path.join(project_path, "configs", ts_yaml)

                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"ts_backbone_yaml not found: {config_path}")

                with open(config_path, "r", encoding="utf-8") as f:
                    cfg_all = yaml.safe_load(f)

                if self.dataset_key not in cfg_all:
                    raise KeyError(f"{self.dataset_key} not found in {config_path}")

                self.ds_cfg = cfg_all[self.dataset_key]

        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 15)))
        self.num_class = int(self.ds_cfg.get("num_labels", getattr(args, "num_class", 12)))
        self.seq_len_orig = int(getattr(args, "seq_len", 200))

        # if hasattr(args, "label_names") and args.label_names is not None:
        #     self.label_names = [str(x) for x in args.label_names]
        # else:
        #     self.label_names = self._get_label_names()

        self.label_names = get_label_names_from_cfg(
            self.ds_cfg,
            num_class=self.num_class,
        )

        print(f"[AlignmentModel] dataset_key={self.dataset_key}")
        print(f"[AlignmentModel] channel_num={self.C}, num_class={self.num_class}")
        print(f"[AlignmentModel] label_names={self.label_names}")

        # ============================================================
        # Load frozen VQ-VAE
        # ============================================================
        self.vq_net = IMU_VQ_Model(args)

        vqvae_key = getattr(args, "vqvae_path", None)
        self.qua_path = None

        if vqvae_key is not None:
            self.qua_path = self.ds_cfg.get(vqvae_key, None)

        if self.qua_path is None:
            self.qua_path = getattr(args, "vqvae_ckpt_dir", None)

        if self.qua_path is None:
            raise ValueError(
                "VQ-VAE checkpoint directory is not provided. "
                "Set args.vqvae_path as a key in ds_cfg, or set args.vqvae_ckpt_dir."
            )

        vq_ckpt_path = os.path.join(self.qua_path, "best_wrapper.pth")

        if not os.path.exists(vq_ckpt_path):
            raise FileNotFoundError(f"VQ-VAE checkpoint not found: {vq_ckpt_path}")

        print(f"[AlignmentModel] Loading VQ-VAE from: {vq_ckpt_path}")

        vq_state = torch.load(vq_ckpt_path, map_location="cpu")

        if isinstance(vq_state, dict) and "state_dict" in vq_state:
            vq_state = vq_state["state_dict"]

        self.vq_net.load_state_dict(vq_state, strict=True)
        self.vq_net.to(self.device)
        self.vq_net.eval()

        for p in self.vq_net.parameters():
            p.requires_grad = False

        # ============================================================
        # VQ primitive setting
        # ============================================================
        self.vq_stride = int(self.vq_net.stride_t ** self.vq_net.down_t)
        self.patch_len = self.vq_stride

        self.seq_len_pad = (
            (self.seq_len_orig + self.vq_stride - 1) // self.vq_stride
        ) * self.vq_stride

        self.P = self.seq_len_pad // self.patch_len

        print(
            f"[AlignmentModel] Orig={self.seq_len_orig}, "
            f"Stride={self.vq_stride}, Padded={self.seq_len_pad}, Patches={self.P}"
        )

        self.num_primitives = int(
            getattr(self.vq_net, "code_num", None)
            or getattr(self.vq_net, "num_code", None)
        )

        if self.num_primitives <= 0:
            raise ValueError(f"Invalid num_primitives={self.num_primitives}")

        self.mask_token_id = self.num_primitives

        # ============================================================
        # Codebook
        # ============================================================
        self.codebook = load_codebook_from_path_or_vq(
            vq_net=self.vq_net,
            ckpt_dir=self.qua_path,
            device=self.device,
        )

        self.num_vq_codes = int(self.codebook.shape[0])
        self.vq_dim = int(self.codebook.shape[1])

        if self.num_vq_codes != self.num_primitives:
            raise ValueError(
                f"Codebook size mismatch: codebook K={self.num_vq_codes}, "
                f"vq_net num_primitives={self.num_primitives}."
            )

        # ============================================================
        # Primitive profile
        # ============================================================
        profile_path = getattr(args, "primitive_profile_path", None)

        if profile_path is None:
            strong_profile_path = os.path.join(self.qua_path, "primitive_profile_strong.json")
            weak_profile_path = os.path.join(self.qua_path, "primitive_profile.json")

            if os.path.exists(strong_profile_path):
                profile_path = strong_profile_path
            elif os.path.exists(weak_profile_path):
                profile_path = weak_profile_path
            else:
                profile_path = None


        if profile_path is not None:
            print(f"[AlignmentModel] Using primitive profile: {profile_path}")
        else:
            print("[AlignmentModel] No primitive profile is provided. Use default primitive descriptions.")

        # ============================================================
        # Teacher core
        # ============================================================
        if not hasattr(args, "llama_name") or args.llama_name is None:
            raise ValueError("args.llama_name is required for AlignmentModel.")

        self.lambda_activity = float(getattr(args, "lambda_activity", 0.5))

        self.teacher_core = PrimitiveLLMTeacherCore(
            llm_path=args.llama_name,
            codebook_weights=self.codebook,
            mask_token_id=self.mask_token_id,
            max_patches=self.P,
            device=self.device,
            temperature=float(getattr(args, "teacher_temperature", 1.0)),
            system_prompt=str(
                getattr(
                    args,
                    "teacher_prompt",
                    "You are a wearable-sensor motion primitive teacher. "
                    "Each primitive represents a local IMU motion pattern with semantic profile information, "
                    "including activity association, motion intensity, temporal variation, periodicity, "
                    "spectral structure, dominant sensor channel, and transition context. "
                    "Given the visible primitive sequence with masked positions, recover the missing primitives "
                    "and infer the human activity.",
                )
            ),
            num_classes=self.num_class,
            label_names=self.label_names,
            primitive_profile_path=profile_path,
            use_semantic_primitive=bool(int(getattr(args, "use_semantic_primitive", 1))),
            semantic_weight=float(getattr(args, "semantic_weight", 0.5)),
            enable_explanation=bool(int(getattr(args, "enable_explanation", 1))),
        )

        self.teacher_core.freeze_llm_only()

    # ============================================================
    # Dataset label names
    # ============================================================
    def _get_label_names(self) -> Optional[List[str]]:
        """
        Get label names from dataset config.

        Priority:
            1. ds_cfg["label_names"] = ["Walking", ...]
            2. ds_cfg["id2label"] = {0: "Walking", 1: "..."}
            3. ds_cfg["id2label"] = {"0": "Walking", "1": "..."}
            4. None
        """
        if not isinstance(self.ds_cfg, dict):
            return None

        # ------------------------------------------------------------
        # 1) Direct label_names list
        # ------------------------------------------------------------
        if "label_names" in self.ds_cfg and self.ds_cfg["label_names"] is not None:
            label_names = [str(x) for x in self.ds_cfg["label_names"]]

            num_class = int(getattr(self, "num_class", len(label_names)))

            if len(label_names) != num_class:
                raise ValueError(
                    f"len(label_names)={len(label_names)} does not match num_class={num_class}."
                )

            return label_names

        # ------------------------------------------------------------
        # 2) id2label dict from YAML
        # ------------------------------------------------------------
        if "id2label" in self.ds_cfg and self.ds_cfg["id2label"] is not None:
            id2label = self.ds_cfg["id2label"]

            if not isinstance(id2label, dict):
                raise TypeError(
                    f"ds_cfg['id2label'] should be dict, got {type(id2label)}"
                )

            # YAML may load keys as int or str. Normalize to int.
            normalized = {}

            for k, v in id2label.items():
                try:
                    kk = int(k)
                except Exception:
                    raise ValueError(f"id2label key should be convertible to int, got {k}")

                normalized[kk] = str(v)

            num_class = int(getattr(self, "num_class", len(normalized)))

            missing = [i for i in range(num_class) if i not in normalized]

            if len(missing) > 0:
                raise ValueError(
                    f"id2label is missing labels for ids: {missing}. "
                    f"Available keys: {sorted(normalized.keys())}"
                )

            label_names = [normalized[i] for i in range(num_class)]

            return label_names

        return None

    # ============================================================
    # Mask sampling
    # ============================================================
    def _sample_alignment_mask(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Sample primitive mask.

        Args:
            valid_mask:
                [B, P], True means valid primitive position.

        Returns:
            final_mask:
                [B, P], True means masked and supervised.
        """
        B, P = valid_mask.shape

        align_mask_rate = getattr(self.args, "align_mask_rate", None)

        if align_mask_rate is not None:
            mask_ratio = float(align_mask_rate)
        else:
            mask_min = float(getattr(self.args, "align_mask_min", 0.15))
            mask_max = float(getattr(self.args, "align_mask_max", 0.50))

            if mask_min < 0 or mask_max > 1 or mask_min > mask_max:
                raise ValueError(
                    f"Invalid mask range: align_mask_min={mask_min}, align_mask_max={mask_max}"
                )

            mask_ratio = torch.empty(1, device=self.device).uniform_(mask_min, mask_max).item()

        rand_mask = torch.rand(B, P, device=self.device) < mask_ratio
        final_mask = rand_mask & valid_mask

        # Guarantee at least one masked position for each valid sample.
        for b in range(B):
            if valid_mask[b].sum() > 0 and final_mask[b].sum() == 0:
                valid_idx = torch.where(valid_mask[b])[0]
                chosen = valid_idx[
                    torch.randint(0, valid_idx.numel(), (1,), device=self.device)
                ]
                final_mask[b, chosen] = True

        return final_mask

    # ============================================================
    # Forward
    # ============================================================
    def forward(
        self,
        x_imu,
        padding_mask=None,
        mode=None,
        labels=None,
        generate_explanation: bool = False,
    ):
        """
        Args:
            x_imu:
                [B, L, C]

            labels:
                [B]

            mode:
                "train" / "eval" / "classify"
                If mode == "classify", no random mask is used.

        Returns:
            loss, logits, metrics
        """
        if not torch.is_tensor(x_imu):
            x_imu = torch.as_tensor(x_imu)

        x_imu = x_imu.to(self.device).float()

        B, L_orig, C = x_imu.shape

        if C != self.C:
            raise ValueError(f"Input channel C={C}, but dataset expects C={self.C}.")

        if labels is not None:
            labels = labels.to(self.device).long()

        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device).bool()

        # ============================================================
        # Pad to fixed length expected by teacher
        # ============================================================
        x_pad, _ = pad_to_multiple(x_imu, self.vq_stride, pad_value=0.0)

        if x_pad.shape[1] != self.seq_len_pad:
            if x_pad.shape[1] > self.seq_len_pad:
                x_pad = x_pad[:, :self.seq_len_pad, :]
                if padding_mask is not None:
                    padding_mask = padding_mask[:, :self.seq_len_pad]
            else:
                pad_len = self.seq_len_pad - x_pad.shape[1]
                pad = x_pad.new_zeros(B, pad_len, C)
                x_pad = torch.cat([x_pad, pad], dim=1)

                if padding_mask is not None:
                    pad_m = torch.zeros((B, pad_len), device=self.device, dtype=torch.bool)
                    padding_mask = torch.cat([padding_mask, pad_m], dim=1)

        # ============================================================
        # VQ primitive ids
        # ============================================================
        with torch.no_grad():
            gt_ids, valid_mask = self.vq_net.get_token_ids_with_mask(
                features=x_pad,
                padding_mask=padding_mask,
            )

        gt_ids = gt_ids.to(self.device).long()
        valid_mask = valid_mask.to(self.device).bool()

        if gt_ids.shape[1] != self.P:
            if gt_ids.shape[1] > self.P:
                gt_ids = gt_ids[:, :self.P]
                valid_mask = valid_mask[:, :self.P]
            else:
                raise ValueError(f"VQ tokens {gt_ids.shape[1]} < expected P={self.P}")

        # ============================================================
        # Mask strategy
        # ============================================================
        classify_only = mode in ["classify", "eval_no_mask", "test"]

        if classify_only:
            primitive_loss_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        else:
            primitive_loss_mask = self._sample_alignment_mask(valid_mask=valid_mask)

        teacher_input_ids = gt_ids.clone()

        # Mask supervised positions.
        teacher_input_ids[primitive_loss_mask] = self.mask_token_id

        # Padding positions are also replaced by mask token,
        # but they are not supervised by primitive loss.
        teacher_input_ids[~valid_mask] = self.mask_token_id

        # ============================================================
        # Teacher forward
        # ============================================================
        loss, teacher_out, metrics = self.teacher_core(
            masked_ids=teacher_input_ids,
            labels=labels,
            target_ids=gt_ids,
            valid_mask=valid_mask,
            return_loss=True,
            primitive_loss_mask=primitive_loss_mask,
            lambda_activity=self.lambda_activity,
            generate_explanation=generate_explanation,
            max_new_tokens=int(getattr(self.args, "max_new_tokens", 96)),
        )

        logits = teacher_out.logits

        # Add extra metrics.
        with torch.no_grad():
            metrics["loss_align"] = metrics["loss_total"]
            metrics["num_masked"] = int(primitive_loss_mask.sum().detach().item())
            metrics["mask_ratio_actual"] = float(
                primitive_loss_mask.sum().detach().item()
                / max(valid_mask.sum().detach().item(), 1)
            )

            if teacher_out.activity_logits is not None:
                metrics["has_activity_logits"] = 1.0
            else:
                metrics["has_activity_logits"] = 0.0

        return loss, logits, metrics

    # ============================================================
    # Inference helper
    # ============================================================
    def classify(self, x_imu, padding_mask=None):
        """
        Clean primitive classification without random mask.

        Important:
            Do NOT use @torch.no_grad() here, because Stage2 training
            needs gradients through teacher_core.projector / query_embed /
            activity_head, etc.
        """
        if not torch.is_tensor(x_imu):
            x_imu = torch.as_tensor(x_imu)

        x_imu = x_imu.to(self.device).float()

        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device).bool()

        B, L_orig, C = x_imu.shape

        if C != self.C:
            raise ValueError(f"Input channel C={C}, but dataset expects C={self.C}.")

        # ============================================================
        # Pad to fixed length expected by teacher
        # ============================================================
        x_pad, _ = pad_to_multiple(x_imu, self.vq_stride, pad_value=0.0)

        if x_pad.shape[1] != self.seq_len_pad:
            if x_pad.shape[1] > self.seq_len_pad:
                x_pad = x_pad[:, :self.seq_len_pad, :]

                if padding_mask is not None:
                    padding_mask = padding_mask[:, :self.seq_len_pad]
            else:
                pad_len = self.seq_len_pad - x_pad.shape[1]
                pad = x_pad.new_zeros(B, pad_len, C)
                x_pad = torch.cat([x_pad, pad], dim=1)

                if padding_mask is not None:
                    pad_m = torch.zeros(
                        (B, pad_len),
                        device=self.device,
                        dtype=torch.bool,
                    )
                    padding_mask = torch.cat([padding_mask, pad_m], dim=1)

        # ============================================================
        # VQ primitive ids: VQ-VAE is frozen, so this part can be no_grad
        # ============================================================
        with torch.no_grad():
            gt_ids, valid_mask = self.vq_net.get_token_ids_with_mask(
                features=x_pad,
                padding_mask=padding_mask,
            )

        gt_ids = gt_ids.to(self.device).long()
        valid_mask = valid_mask.to(self.device).bool()

        if gt_ids.shape[1] != self.P:
            if gt_ids.shape[1] > self.P:
                gt_ids = gt_ids[:, :self.P]
                valid_mask = valid_mask[:, :self.P]
            else:
                raise ValueError(f"VQ tokens {gt_ids.shape[1]} < expected P={self.P}")

        input_ids = gt_ids.clone()
        input_ids[~valid_mask] = self.mask_token_id

        # ============================================================
        # Teacher classification forward
        # This part must keep grad during training.
        # ============================================================
        out = self.teacher_core(
            masked_ids=input_ids,
            valid_mask=valid_mask,
            return_loss=False,
            generate_explanation=False,
        )

        if out.activity_logits is None:
            raise RuntimeError(
                "teacher_core returned None activity_logits. "
                "Please check whether activity_head is enabled."
            )

        return out.activity_logits, out.activity_probs
    # ============================================================
    # Save / Load
    # ============================================================
    def save_wrapper(self, path):
        self.teacher_core.save_adapter(path)
        print(f"[AlignmentModel] Adapters saved to: {path}")

    def load_wrapper(self, path, map_location="cpu"):
        self.teacher_core.load_adapter(path, strict=True)
        print(f"[AlignmentModel] Loaded alignment adapter from: {path}")
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
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

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
    parser.add_argument('--freeze_llm', type=int, default=1, help='freeze_llm')
    parser.add_argument('--trainable_modules', type=str, default="sensor_proj,channel_id,recon_head,cls_head",
                        help='trainable_modules')
    parser.add_argument('--run_id', type=str, default="202501220_0956", help='trainable_modules')
    parser.add_argument('--debug_fake_llm', type=bool, default=False, help='debug_fake_llm')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    configs = get_configs()
    configs.ts_backbone_yaml = r"D:\fuy\MyCode\SensorLLMLib_v2\configs\ts_backbone.yaml"
    configs.debug_fake_llm = True

    configs.stage=2
    if torch.cuda.is_available() and configs.use_gpu:
        configs.device = torch.device('cuda:{}'.format(configs.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            configs.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            configs.device = torch.device("cpu")
        print('Using cpu or mps')
    model = AlignmentModel(configs).to("cuda:0")
    x = torch.rand(32,96,15)
    c = model(x.to("cuda:0"),None,None)
    d = 'end'