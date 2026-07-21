# models_new_version_run/AlignmentModel.py

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

# ============================================================
# Important:
# Use the SAME teacher implementation as the distillation model.
# Do NOT import PrimitiveLlmTeacher here.
# ============================================================
from models_new_version_run.LLMTeacher import (
    PrimitiveLLMTeacherCore,
    load_codebook_from_path_or_vq,
    pad_to_multiple,
)


class AlignmentModel(nn.Module):
    """
    Restored alignment model.

    This version follows the original distillation-compatible design:

        Frozen VQ-VAE:
            IMU sequence -> VQ primitive ids

        Trainable LLM teacher adapter:
            masked primitive ids -> recover missing primitive ids

    Objective:
        loss = CE(teacher_logits[masked_positions], gt_ids[masked_positions])

    Frozen:
        - VQ-VAE
        - LLM backbone

    Trainable:
        - projector
        - mask_embed_llama
        - primitive_pos_embed
        - query_embed
        - output_head

    This saved adapter is compatible with:

        from models_new_version_run.LLMTeacher import PrimitiveLLMTeacherCore

    in the student distillation model.
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
                    project_path = os.path.abspath(
                        os.path.dirname(os.path.dirname(__file__))
                    )
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

        print(f"[AlignmentModel] dataset_key={self.dataset_key}")
        print(f"[AlignmentModel] channel_num={self.C}, num_class={self.num_class}")

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

        print(f"[AlignmentModel] num_primitives={self.num_primitives}")
        print(f"[AlignmentModel] mask_token_id={self.mask_token_id}")

        # ============================================================
        # Load codebook
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
                f"vq_net num_primitives={self.num_primitives}. "
                "Please check whether best_codebook.pth matches best_wrapper.pth."
            )

        print(
            f"[AlignmentModel] codebook shape: "
            f"K={self.num_vq_codes}, D={self.vq_dim}"
        )

        # ============================================================
        # LLM teacher core
        # ============================================================
        if not hasattr(args, "llama_name") or args.llama_name is None:
            raise ValueError("args.llama_name is required for AlignmentModel.")

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
                    "Recover masked sensor primitives:",
                )
            ),
        )

        # Freeze only LLM backbone.
        # Train projector / mask embedding / position embedding / query embedding / output head.
        self.teacher_core.freeze_llm_only()

        print("[AlignmentModel] Teacher core initialized with LLMTeacher.PrimitiveLLMTeacherCore.")
        print("[AlignmentModel] Alignment objective: masked primitive recovery only.")

    # ============================================================
    # Optional label-name helper
    # This function is kept only for backward compatibility.
    # It is NOT used in the restored alignment model.
    # ============================================================
    def _get_label_names(self) -> Optional[List[str]]:
        if not isinstance(self.ds_cfg, dict):
            return None

        if "label_names" in self.ds_cfg and self.ds_cfg["label_names"] is not None:
            label_names = [str(x) for x in self.ds_cfg["label_names"]]
            num_class = int(getattr(self, "num_class", len(label_names)))

            if len(label_names) != num_class:
                raise ValueError(
                    f"len(label_names)={len(label_names)} does not match num_class={num_class}."
                )

            return label_names

        if "id2label" in self.ds_cfg and self.ds_cfg["id2label"] is not None:
            id2label = self.ds_cfg["id2label"]

            if not isinstance(id2label, dict):
                raise TypeError(
                    f"ds_cfg['id2label'] should be dict, got {type(id2label)}"
                )

            normalized = {}

            for k, v in id2label.items():
                try:
                    kk = int(k)
                except Exception:
                    raise ValueError(
                        f"id2label key should be convertible to int, got {k}"
                    )

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
                    f"Invalid mask range: align_mask_min={mask_min}, "
                    f"align_mask_max={mask_max}"
                )

            mask_ratio = torch.empty(
                1,
                device=self.device,
            ).uniform_(mask_min, mask_max).item()

        rand_mask = torch.rand(B, P, device=self.device) < mask_ratio
        final_mask = rand_mask & valid_mask

        # Guarantee at least one masked valid position for each sample.
        for b in range(B):
            if valid_mask[b].sum() > 0 and final_mask[b].sum() == 0:
                valid_idx = torch.where(valid_mask[b])[0]

                chosen = valid_idx[
                    torch.randint(
                        0,
                        valid_idx.numel(),
                        (1,),
                        device=self.device,
                    )
                ]

                final_mask[b, chosen] = True

        return final_mask

    def _build_valid_mask_from_length(self, B: int, L_orig: int) -> torch.Tensor:
        """
        Fallback valid mask if VQ-VAE does not provide get_token_ids_with_mask.

        Args:
            B:
                batch size
            L_orig:
                original unpadded sequence length

        Returns:
            valid_mask:
                [B, P]
        """
        valid_patches = (L_orig + self.patch_len - 1) // self.patch_len
        valid_patches = min(valid_patches, self.P)

        valid_mask = torch.zeros(
            (B, self.P),
            device=self.device,
            dtype=torch.bool,
        )

        valid_mask[:, :valid_patches] = True

        return valid_mask

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
        Alignment forward.

        Args:
            x_imu:
                [B, L, C]

            padding_mask:
                Optional [B, L], True means valid if your VQ-VAE uses this convention.
                This follows your original get_token_ids_with_mask usage.

            mode:
                "train" / "eval" / "classify" / "test"

            labels:
                Ignored. Kept only for compatibility with your training framework.

            generate_explanation:
                Ignored. Kept only for compatibility.

        Returns:
            loss:
                primitive reconstruction CE loss

            logits:
                teacher primitive logits, [B, P, K]

            metrics:
                dict
        """
        if not torch.is_tensor(x_imu):
            x_imu = torch.as_tensor(x_imu)

        x_imu = x_imu.to(self.device).float()

        B, L_orig, C = x_imu.shape

        if C != self.C:
            raise ValueError(f"Input channel C={C}, but dataset expects C={self.C}.")

        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device).bool()

        # ============================================================
        # Optional instance normalization.
        # Keep it consistent with your distillation model.
        # Use only if VQ-VAE was trained with the same normalization.
        # ============================================================
        if bool(getattr(self.args, "use_instance_norm", False)):
            eps = float(getattr(self.args, "norm_eps", 1e-5))
            mu = x_imu.mean(dim=1, keepdim=True)
            sigma = x_imu.std(dim=1, keepdim=True).clamp_min(eps)
            x_imu = (x_imu - mu) / sigma

        # ============================================================
        # Pad to fixed length expected by VQ-VAE and teacher
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
        # Frozen VQ-VAE: IMU -> primitive ids
        # ============================================================
        with torch.no_grad():
            if hasattr(self.vq_net, "get_token_ids_with_mask"):
                gt_ids, valid_mask = self.vq_net.get_token_ids_with_mask(
                    features=x_pad,
                    padding_mask=padding_mask,
                )
            else:
                gt_ids = self.vq_net.get_token_ids(x_pad)
                valid_mask = self._build_valid_mask_from_length(
                    B=B,
                    L_orig=L_orig,
                )

        gt_ids = gt_ids.to(self.device).long()
        valid_mask = valid_mask.to(self.device).bool()

        if gt_ids.shape[1] != self.P:
            if gt_ids.shape[1] > self.P:
                gt_ids = gt_ids[:, :self.P]
                valid_mask = valid_mask[:, :self.P]
            else:
                raise ValueError(
                    f"VQ tokens {gt_ids.shape[1]} < expected P={self.P}."
                )

        # ============================================================
        # Mask strategy
        # ============================================================
        no_mask_mode = mode in ["classify", "eval_no_mask", "test"]

        if no_mask_mode:
            primitive_loss_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        else:
            primitive_loss_mask = self._sample_alignment_mask(
                valid_mask=valid_mask,
            )

        teacher_input_ids = gt_ids.clone()

        # Supervised masked positions.
        teacher_input_ids[primitive_loss_mask] = self.mask_token_id

        # Padding positions should not be real teacher context.
        teacher_input_ids[~valid_mask] = self.mask_token_id

        # ============================================================
        # Teacher forward
        # ============================================================
        teacher_out = self.teacher_core(teacher_input_ids)

        logits = teacher_out.logits
        probs = teacher_out.probs

        # ============================================================
        # Primitive recovery loss
        # ============================================================
        pred_masked = logits[primitive_loss_mask]
        target_masked = gt_ids[primitive_loss_mask]

        if target_masked.numel() > 0:
            loss_primitive = F.cross_entropy(
                pred_masked,
                target_masked,
            )

            with torch.no_grad():
                pred_ids = pred_masked.argmax(dim=-1)
                primitive_acc = (pred_ids == target_masked).float().mean()
        else:
            loss_primitive = torch.tensor(
                0.0,
                device=self.device,
                requires_grad=True,
            )

            primitive_acc = torch.tensor(
                0.0,
                device=self.device,
            )

        loss = loss_primitive

        # ============================================================
        # Metrics
        # ============================================================
        with torch.no_grad():
            num_valid = int(valid_mask.sum().detach().item())
            num_masked = int(primitive_loss_mask.sum().detach().item())

            if target_masked.numel() > 0:
                masked_probs = probs[primitive_loss_mask]

                teacher_conf = masked_probs.max(dim=-1).values.mean()

                teacher_entropy = -(
                    masked_probs * (masked_probs + 1e-8).log()
                ).sum(dim=-1).mean()

                teacher_entropy_norm = teacher_entropy / torch.log(
                    torch.tensor(
                        float(self.num_primitives),
                        device=self.device,
                    )
                )
            else:
                teacher_conf = torch.tensor(0.0, device=self.device)
                teacher_entropy = torch.tensor(0.0, device=self.device)
                teacher_entropy_norm = torch.tensor(0.0, device=self.device)

            metrics = {
                "loss_total": float(loss.detach().item()),
                "loss_align": float(loss.detach().item()),
                "loss_primitive": float(loss_primitive.detach().item()),
                "loss_recon": float(loss_primitive.detach().item()),

                # Backward-compatible metric names.
                "primitive_acc": float(primitive_acc.detach().item()),
                "mask_acc": float(primitive_acc.detach().item()),

                "teacher_conf": float(teacher_conf.detach().item()),
                "teacher_entropy": float(teacher_entropy.detach().item()),
                "teacher_entropy_norm": float(teacher_entropy_norm.detach().item()),

                "num_valid": num_valid,
                "num_masked": num_masked,
                "mask_ratio_actual": float(
                    num_masked / max(num_valid, 1)
                ),

                # Keep this key for compatibility with old logs.
                # Restored teacher has no activity head.
                "has_activity_logits": 0.0,
            }

        return loss, logits, metrics

    # ============================================================
    # Inference helper
    # ============================================================
    def predict_primitives(
        self,
        x_imu,
        padding_mask=None,
        mask_bool=None,
    ):
        """
        Optional helper for inspecting teacher primitive predictions.

        This does NOT perform activity classification.

        Args:
            x_imu:
                [B, L, C]

            mask_bool:
                Optional [B, P], True means masked.
                If None, no valid primitive is masked.

        Returns:
            logits:
                [B, P, K]

            probs:
                [B, P, K]

            gt_ids:
                [B, P]

            valid_mask:
                [B, P]
        """
        if not torch.is_tensor(x_imu):
            x_imu = torch.as_tensor(x_imu)

        x_imu = x_imu.to(self.device).float()

        B, L_orig, C = x_imu.shape

        if C != self.C:
            raise ValueError(f"Input channel C={C}, but dataset expects C={self.C}.")

        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device).bool()

        if bool(getattr(self.args, "use_instance_norm", False)):
            eps = float(getattr(self.args, "norm_eps", 1e-5))
            mu = x_imu.mean(dim=1, keepdim=True)
            sigma = x_imu.std(dim=1, keepdim=True).clamp_min(eps)
            x_imu = (x_imu - mu) / sigma

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

        with torch.no_grad():
            if hasattr(self.vq_net, "get_token_ids_with_mask"):
                gt_ids, valid_mask = self.vq_net.get_token_ids_with_mask(
                    features=x_pad,
                    padding_mask=padding_mask,
                )
            else:
                gt_ids = self.vq_net.get_token_ids(x_pad)
                valid_mask = self._build_valid_mask_from_length(
                    B=B,
                    L_orig=L_orig,
                )

        gt_ids = gt_ids.to(self.device).long()
        valid_mask = valid_mask.to(self.device).bool()

        if gt_ids.shape[1] != self.P:
            if gt_ids.shape[1] > self.P:
                gt_ids = gt_ids[:, :self.P]
                valid_mask = valid_mask[:, :self.P]
            else:
                raise ValueError(
                    f"VQ tokens {gt_ids.shape[1]} < expected P={self.P}."
                )

        input_ids = gt_ids.clone()

        if mask_bool is not None:
            mask_bool = mask_bool.to(self.device).bool()

            if mask_bool.shape != input_ids.shape:
                raise ValueError(
                    f"mask_bool shape should be {tuple(input_ids.shape)}, "
                    f"got {tuple(mask_bool.shape)}"
                )

            input_ids[mask_bool & valid_mask] = self.mask_token_id

        input_ids[~valid_mask] = self.mask_token_id

        teacher_out = self.teacher_core(input_ids)

        return teacher_out.logits, teacher_out.probs, gt_ids, valid_mask

    def classify(self, x_imu, padding_mask=None):
        """
        Restored alignment model does not perform HAR classification.

        HAR classification should be done by the student Stage-2 model.
        """
        raise NotImplementedError(
            "The restored AlignmentModel only performs masked primitive recovery. "
            "It does not perform activity classification. "
            "Use the student Stage-2 model for HAR classification."
        )

    # ============================================================
    # Save / Load
    # ============================================================
    def save_wrapper(self, path):
        """
        Save only the LLM teacher adapter.

        Saved keys are compatible with LLMTeacher.PrimitiveLLMTeacherCore.load_adapter():

            - projector
            - output_head
            - mask_embed_llama
            - primitive_pos_embed
            - query_embed
            - meta
        """
        self.teacher_core.save_adapter(path)
        print(f"[AlignmentModel] Adapter saved to: {path}")

    def load_wrapper(self, path, map_location="cpu"):
        """
        Load LLM teacher adapter.
        """
        load_info = self.teacher_core.load_adapter(path, strict=True)
        print(f"[AlignmentModel] Loaded alignment adapter from: {path}")
        print(f"[AlignmentModel] load info: {load_info}")
        return load_info