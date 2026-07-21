# models_new_version_run/SensorLLMFuy_test_withllm_mae_vqvae.py
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml
except Exception:
    yaml = None

from models_new_version_run.VQ_VAE import IMU_VQ_Model
from models_new_version_run.LLMTeacher import (
    PrimitiveLLMTeacherCore,
    pad_to_multiple,
    extract_codebook,
    load_codebook_from_path_or_vq,
)


# ============================================================
# Student Encoder
# ============================================================
class PatchEmbeddingConv(nn.Module):
    """
    Convolutional patch embedding for IMU sequences.

    Input:
        x: [B, L_pad, C]

    Output:
        h: [B, P, D]
    """

    def __init__(
        self,
        seq_len_pad: int,
        patch_len: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()

        self.seq_len = int(seq_len_pad)
        self.patch_len = int(patch_len)
        self.num_patches = self.seq_len // self.patch_len
        self.embed_dim = int(embed_dim)

        if self.seq_len % self.patch_len != 0:
            raise ValueError(
                f"seq_len_pad must be divisible by patch_len, got "
                f"seq_len_pad={self.seq_len}, patch_len={self.patch_len}"
            )

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim // 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )

        self.proj = nn.Linear(self.patch_len * embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask_bool: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, C = x.shape

        if L != self.seq_len:
            raise ValueError(f"Expected L={self.seq_len}, but got L={L}.")

        # [B, L, C] -> [B, C, L] -> [B, D, L] -> [B, L, D]
        h = self.stem(x.transpose(1, 2)).transpose(1, 2)

        # [B, L, D] -> [B, P, patch_len * D]
        h = h.reshape(B, self.num_patches, self.patch_len, self.embed_dim)
        h = h.reshape(B, self.num_patches, self.patch_len * self.embed_dim)

        h = self.proj(h)
        h = self.norm(h)
        h = h + self.pos_embed

        if mask_bool is not None:
            if mask_bool.shape != (B, self.num_patches):
                raise ValueError(
                    f"mask_bool shape should be {(B, self.num_patches)}, "
                    f"got {tuple(mask_bool.shape)}"
                )

            w = mask_bool.unsqueeze(-1).type_as(h)
            mask_tokens = self.mask_token.expand(B, self.num_patches, self.embed_dim)

            h = h * (1.0 - w) + mask_tokens * w

        return h


class StrongStudent(nn.Module):
    """
    Student encoder for primitive prediction and downstream classification.
    """

    def __init__(
        self,
        seq_len_pad: int,
        patch_len: int,
        in_channels: int,
        dim_model: int,
        num_vq_codes: int,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbeddingConv(
            seq_len_pad=seq_len_pad,
            patch_len=patch_len,
            in_channels=in_channels,
            embed_dim=dim_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=nhead,
            dim_feedforward=dim_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=dim_model,
        #     nhead=nhead,
        #     dim_feedforward=dim_model * 4,
        #     dropout=dropout,
        #     batch_first=True,
        #     norm_first=True,
        # )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.vocab_head = nn.Linear(dim_model, num_vq_codes)

    def forward(
        self,
        x: torch.Tensor,
        mask_bool: Optional[torch.Tensor] = None,
    ):
        emb = self.patch_embed(x, mask_bool)
        feat = self.transformer(emb)
        logits = self.vocab_head(feat)

        return logits, feat


# ============================================================
# Main Model
# ============================================================
class Model(nn.Module):
    """
    ReasonHAR model.

    Stage 1:
        - Frozen VQ-VAE generates discrete primitive targets.
        - Student predicts masked primitives.
        - Optional frozen LLM teacher provides soft primitive distributions.

    Stage 2:
        - Teacher is removed.
        - Student encoder + attention pooling + classifier are used for HAR.
    """

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.stage = int(getattr(args, "stage", 1))
        self.device = args.device

        print(f"[Model] Init in Stage: {self.stage}")

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

        # ============================================================
        # VQ-VAE loading
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

        print(f"[Model] Loading VQ-VAE from: {vq_ckpt_path}")

        vq_state = torch.load(vq_ckpt_path, map_location="cpu")

        if isinstance(vq_state, dict) and "state_dict" in vq_state:
            vq_state = vq_state["state_dict"]

        self.vq_net.load_state_dict(vq_state, strict=True)
        self.vq_net.eval()

        for p in self.vq_net.parameters():
            p.requires_grad = False

        # ============================================================
        # VQ-token / patch alignment
        # ============================================================
        self.vq_stride = int(self.vq_net.stride_t ** self.vq_net.down_t)
        self.patch_len = self.vq_stride

        self.seq_len_pad = (
            (self.seq_len_orig + self.vq_stride - 1) // self.vq_stride
        ) * self.vq_stride

        self.P = self.seq_len_pad // self.patch_len

        print(
            f"[Model Alignment] Orig={self.seq_len_orig}, "
            f"Stride={self.vq_stride} -> Padded={self.seq_len_pad}, Patches={self.P}"
        )

        self.num_primitives = int(
            getattr(self.vq_net, "code_num", None) or getattr(self.vq_net, "num_code", None)
        )

        if self.num_primitives <= 0:
            raise ValueError(f"Invalid num_primitives={self.num_primitives}")

        self.mask_token_id = self.num_primitives

        # ============================================================
        # Student
        # ============================================================
        self.dim_student = int(getattr(args, "dim_student", 256))
        self.student_nhead = int(getattr(args, "student_nhead", 4))
        self.student_num_layers = int(getattr(args, "student_num_layers", 4))
        self.student_dropout = float(getattr(args, "student_dropout", 0.1))

        self.student = StrongStudent(
            seq_len_pad=self.seq_len_pad,
            patch_len=self.patch_len,
            in_channels=self.C,
            dim_model=self.dim_student,
            num_vq_codes=self.num_primitives,
            nhead=self.student_nhead,
            num_layers=self.student_num_layers,
            dropout=self.student_dropout,
        ).to(self.device)

        self.mask_rate = float(getattr(args, "mask_rate", 0.4))
        self.lambda_distill = float(getattr(args, "lambda_distill", 0.0))

        if self.mask_rate < 0 or self.mask_rate > 1:
            raise ValueError(f"mask_rate should be in [0, 1], got {self.mask_rate}")

        # ============================================================
        # Optional LLM teacher for Stage 1
        # ============================================================
        self.teacher = None

        if self.stage == 1 and self.lambda_distill > 0:
            if not hasattr(args, "llama_name") or args.llama_name is None:
                raise ValueError("args.llama_name is required when lambda_distill > 0.")

            codebook_weights = load_codebook_from_path_or_vq(
                vq_net=self.vq_net,
                ckpt_dir=self.qua_path,
                device=self.device,
            )

            if codebook_weights.shape[0] != self.num_primitives:
                raise ValueError(
                    f"Codebook size mismatch: codebook K={codebook_weights.shape[0]}, "
                    f"vq_net num_primitives={self.num_primitives}. "
                    "Please check whether best_codebook.pth matches best_wrapper.pth."
                )

            alignment_path = self.ds_cfg.get("alignment_path", None)
            adapter_path = getattr(args, "teacher_adapter_path", None)

            if adapter_path is None and alignment_path is not None:
                candidate = os.path.join(alignment_path, "best_wrapper.pth")
                if os.path.exists(candidate):
                    adapter_path = candidate

            allow_random_teacher = bool(getattr(args, "allow_random_teacher", False))

            if adapter_path is None and not allow_random_teacher:
                raise FileNotFoundError(
                    "Teacher adapter is required when lambda_distill > 0. "
                    "Set args.teacher_adapter_path or ds_cfg['alignment_path']/best_wrapper.pth. "
                    "For CE-only pretraining, set lambda_distill=0."
                )

            self.teacher = PrimitiveLLMTeacherCore(
                llm_path=args.llama_name,
                codebook_weights=codebook_weights,
                mask_token_id=self.mask_token_id,
                max_patches=self.P,
                device=self.device,
                temperature=float(getattr(args, "distill_temperature", 1.0)),
                system_prompt=str(
                    getattr(args, "teacher_prompt", "Recover masked sensor primitives:")
                ),
            )

            if adapter_path is not None and os.path.exists(adapter_path):
                load_info = self.teacher.load_adapter(adapter_path, strict=True)
                print(f"[Model] Loaded teacher adapter from: {adapter_path}")
                print(f"[Model] teacher load info: {load_info}")
            elif allow_random_teacher:
                print("[Model][WARNING] Using randomly initialized teacher adapter.")
            else:
                raise FileNotFoundError(f"Teacher adapter not found: {adapter_path}")

            self.teacher.freeze_all()

        # ============================================================
        # Stage 2 classification head
        # ============================================================
        if self.stage == 2:
            self.pool_query = nn.Parameter(
                torch.randn(1, 1, self.dim_student) * 0.02
            )

            self.pool_attn = nn.MultiheadAttention(
                embed_dim=self.dim_student,
                num_heads=self.student_nhead,
                dropout=self.student_dropout,
                batch_first=True,
            )

            self.classifier = nn.Sequential(
                nn.LayerNorm(self.dim_student),
                nn.Linear(self.dim_student, self.dim_student),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.dim_student, self.num_class),
            )

    # ============================================================
    # Masking
    # ============================================================
    def get_mask(
        self,
        B: int,
        P: int,
        mask_ratio: float,
        mode: str = "random",
    ) -> torch.Tensor:
        """
        Return token-level mask [B, P].
        True means masked.
        """
        mode = str(mode).lower()

        if mode == "random":
            return torch.rand(B, P, device=self.device) < mask_ratio

        if mode == "block":
            mask = torch.zeros((B, P), device=self.device, dtype=torch.bool)

            block_len = max(1, int(P * mask_ratio))
            block_len = min(block_len, P)

            for i in range(B):
                start = torch.randint(0, P - block_len + 1, (1,), device=self.device).item()
                mask[i, start:start + block_len] = True

            return mask

        raise ValueError(
            f"Unknown mask_mode={mode}. "
            "Use 'random' for main experiments or 'block' for ablation. "
            "Do not use channel masking in Stage 1 because it corrupts VQ targets."
        )

    def _build_loss_valid_mask(self, B: int, L_orig: int) -> torch.Tensor:
        valid_patches = (L_orig + self.patch_len - 1) // self.patch_len
        valid_patches = min(valid_patches, self.P)

        loss_valid_mask = torch.zeros((B, self.P), device=self.device, dtype=torch.bool)
        loss_valid_mask[:, :valid_patches] = True

        return loss_valid_mask

    def _ensure_at_least_one_masked_valid(
        self,
        mask_bool: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        final_mask = mask_bool & valid_mask

        if final_mask.sum() == 0:
            B = mask_bool.shape[0]
            for b in range(B):
                valid_idx = torch.where(valid_mask[b])[0]
                if valid_idx.numel() > 0:
                    mask_bool[b, valid_idx[0]] = True

        return mask_bool

    # ============================================================
    # Forward
    # ============================================================
    def forward(self, x_imu, padding_mask=None, mode=None, labels=None):
        if mode is None:
            mode = "pretrain" if self.stage == 1 else "classify"

        if not torch.is_tensor(x_imu):
            x_imu = torch.as_tensor(x_imu)

        x_imu = x_imu.to(self.device).float()

        B, L_orig, C = x_imu.shape

        if C != self.C:
            raise ValueError(f"Input channel C={C}, but model expects C={self.C}.")

        # Optional instance normalization.
        # Use only if VQ-VAE was trained with the same normalization.
        if bool(getattr(self.args, "use_instance_norm", False)):
            eps = float(getattr(self.args, "norm_eps", 1e-5))
            mu = x_imu.mean(dim=1, keepdim=True)
            sigma = x_imu.std(dim=1, keepdim=True).clamp_min(eps)
            x_imu = (x_imu - mu) / sigma

        x_pad, _ = pad_to_multiple(x_imu, self.vq_stride, pad_value=0.0)

        if x_pad.shape[1] != self.seq_len_pad:
            if x_pad.shape[1] > self.seq_len_pad:
                x_pad = x_pad[:, :self.seq_len_pad, :]
            else:
                pad_len = self.seq_len_pad - x_pad.shape[1]
                pad = x_pad.new_zeros(B, pad_len, C)
                x_pad = torch.cat([x_pad, pad], dim=1)

        loss_valid_mask = self._build_loss_valid_mask(B=B, L_orig=L_orig)

        # ========================================================
        # Stage 1: masked primitive prediction + optional distill
        # ========================================================
        if self.stage == 1:
            mask_mode = str(getattr(self.args, "mask_mode", "random")).lower()

            if mask_mode == "channel":
                raise NotImplementedError(
                    "Stage-1 channel masking is disabled because it corrupts VQ targets. "
                    "Use mask_mode='random' for main distillation. "
                    "Evaluate channel-wise missingness at test time."
                )

            with torch.no_grad():
                gt_ids = self.vq_net.get_token_ids(x_pad)

            if gt_ids.shape[1] != self.P:
                if gt_ids.shape[1] > self.P:
                    gt_ids = gt_ids[:, :self.P]
                else:
                    raise ValueError(f"VQ tokens {gt_ids.shape[1]} < student patches {self.P}.")

            gt_ids = gt_ids.to(self.device).long()

            mask_bool = self.get_mask(
                B=B,
                P=self.P,
                mask_ratio=self.mask_rate,
                mode=mask_mode,
            )

            mask_bool = self._ensure_at_least_one_masked_valid(
                mask_bool=mask_bool,
                valid_mask=loss_valid_mask,
            )

            final_mask = mask_bool & loss_valid_mask

            student_logits, _ = self.student(x_pad, mask_bool)  # [B, P, K]

            target_masked = gt_ids[final_mask]
            pred_masked = student_logits[final_mask]

            if target_masked.numel() > 0:
                loss_prim = F.cross_entropy(pred_masked, target_masked)

                with torch.no_grad():
                    student_pred = pred_masked.argmax(dim=-1)
                    mask_acc = (student_pred == target_masked).float().mean()
            else:
                loss_prim = torch.tensor(0.0, device=self.device, requires_grad=True)
                mask_acc = torch.tensor(0.0, device=self.device)

            loss_distill = torch.tensor(0.0, device=self.device)
            teacher_entropy = torch.tensor(0.0, device=self.device)
            teacher_entropy_norm = torch.tensor(0.0, device=self.device)
            teacher_acc = torch.tensor(0.0, device=self.device)
            teacher_conf = torch.tensor(0.0, device=self.device)
            teacher_conf_raw = torch.tensor(0.0, device=self.device)

            if self.teacher is not None and self.lambda_distill > 0:
                teacher_input_ids = gt_ids.clone()

                # Important:
                # Padding patches should not be treated as real teacher context.
                teacher_input_ids[mask_bool | (~loss_valid_mask)] = self.mask_token_id

                with torch.no_grad():
                    teacher_out = self.teacher(teacher_input_ids)

                t_probs_temp = teacher_out.probs[final_mask].detach()

                if pred_masked.numel() > 0 and t_probs_temp.numel() > 0:
                    use_hard_label = int(getattr(self.args, "use_hard_label", 0))
                    distill_temperature = float(getattr(self.args, "distill_temperature", 1.0))

                    if distill_temperature <= 0:
                        raise ValueError(
                            f"distill_temperature must be > 0, got {distill_temperature}"
                        )

                    with torch.no_grad():
                        t_logits_raw = teacher_out.logits[final_mask].detach()
                        t_probs_raw = F.softmax(t_logits_raw, dim=-1)

                        teacher_entropy = -(
                            t_probs_temp * (t_probs_temp + 1e-8).log()
                        ).sum(dim=-1).mean()

                        teacher_entropy_norm = teacher_entropy / torch.log(
                            torch.tensor(float(self.num_primitives), device=self.device)
                        )

                        teacher_conf = t_probs_temp.max(dim=-1).values.mean()
                        teacher_conf_raw = t_probs_raw.max(dim=-1).values.mean()

                        teacher_pred = t_logits_raw.argmax(dim=-1)
                        teacher_acc = (teacher_pred == target_masked).float().mean()

                    if use_hard_label:
                        teacher_hard_labels = t_logits_raw.argmax(dim=-1)

                        loss_distill = F.cross_entropy(
                            pred_masked,
                            teacher_hard_labels,
                            reduction="mean",
                        )
                    else:
                        # Recompute both sides with the same distillation temperature.
                        t_probs_for_kl = F.softmax(
                            t_logits_raw / distill_temperature,
                            dim=-1,
                        )

                        s_log_probs_for_kl = F.log_softmax(
                            pred_masked / distill_temperature,
                            dim=-1,
                        )

                        loss_distill = F.kl_div(
                            s_log_probs_for_kl,
                            t_probs_for_kl,
                            reduction="batchmean",
                        ) * (distill_temperature ** 2)

            loss_total = loss_prim + self.lambda_distill * loss_distill

            return loss_total, student_logits, {
                "loss_prim": float(loss_prim.detach().item()),
                "loss_recon": float(loss_prim.detach().item()),  # backward-compatible key
                "loss_distill": float(loss_distill.detach().item()),
                "loss_total": float(loss_total.detach().item()),
                "mask_acc": float(mask_acc.detach().item()),
                "teacher_acc": float(teacher_acc.detach().item()),
                "teacher_conf": float(teacher_conf.detach().item()),
                "teacher_conf_raw": float(teacher_conf_raw.detach().item()),
                "teacher_entropy": float(teacher_entropy.detach().item()),
                "teacher_entropy_norm": float(teacher_entropy_norm.detach().item()),
            }

        # ========================================================
        # Stage 2: classification
        # ========================================================
        if self.stage == 2:
            _, feat = self.student(x_pad, mask_bool=None)  # [B, P, D]

            valid_patches = int(loss_valid_mask[0].sum().item())
            valid_patches = max(1, min(valid_patches, self.P))

            feat_valid = feat[:, :valid_patches, :]  # [B, Pv, D]

            query = self.pool_query.expand(B, -1, -1)  # [B, 1, D]
            pooled, _ = self.pool_attn(query, feat_valid, feat_valid)

            global_feat = pooled.squeeze(1)
            logits = self.classifier(global_feat)

            return logits

        raise ValueError(f"Unknown stage: {self.stage}")

    # ============================================================
    # Save / load student wrapper
    # ============================================================
    def save_wrapper(self, path):
        """
        Save student and stage-related lightweight modules.
        Exclude:
            - frozen VQ-VAE
            - frozen teacher
        """
        sd = self.state_dict()
        to_save = {}

        for k, v in sd.items():
            if k.startswith("vq_net.") or k.startswith("teacher."):
                continue
            if "vq_net" in k or "teacher" in k:
                continue

            to_save[k] = v.detach().cpu()

        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(to_save, path)
        print(f"[save_wrapper] Saved model to: {path}")

    def load_wrapper(self, path, map_location="cpu"):
        """
        Load student pretraining checkpoint into Stage 2 model.
        Missing classifier / pooling keys are expected when loading Stage 1 into Stage 2.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        sd = torch.load(path, map_location=map_location)

        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        sd = {
            k: v for k, v in sd.items()
            if "vq_net" not in k and "teacher" not in k
        }

        ret = self.load_state_dict(sd, strict=False)

        print("[load_wrapper] Loaded student weights.")
        print("[load_wrapper] missing keys:", ret.missing_keys)
        print("[load_wrapper] unexpected keys:", ret.unexpected_keys)