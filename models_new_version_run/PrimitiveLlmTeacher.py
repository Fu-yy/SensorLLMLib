import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False
    AutoTokenizer = None
    AutoModelForCausalLM = None


# ============================================================
# CUDA attention compatibility
# ============================================================
if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


# ============================================================
# Output structures
# ============================================================
@dataclass
class TeacherOut:
    """
    Backward-compatible teacher output.

    logits/probs:
        Primitive recovery logits/probs, shape [B, P, K].

    activity_logits/activity_probs:
        Activity classification logits/probs, shape [B, num_classes].

    explanation_texts:
        Optional generated textual explanations.
    """
    logits: torch.Tensor
    probs: torch.Tensor
    activity_logits: Optional[torch.Tensor] = None
    activity_probs: Optional[torch.Tensor] = None
    explanation_texts: Optional[List[str]] = None


# ============================================================
# Utilities
# ============================================================
def pad_to_multiple(x: torch.Tensor, multiple: int, pad_value: float = 0.0):
    """
    Pad sequence length to a multiple of VQ stride.

    Args:
        x: [B, L, C]
        multiple: temporal stride

    Returns:
        x_pad: [B, L_pad, C]
        L_orig: original sequence length
    """
    if x.dim() != 3:
        raise ValueError(f"x should be [B, L, C], got {tuple(x.shape)}")

    B, L, C = x.shape
    L_pad = ((L + multiple - 1) // multiple) * multiple

    if L_pad == L:
        return x, L

    pad_len = L_pad - L
    pad = x.new_full((B, pad_len, C), pad_value)
    return torch.cat([x, pad], dim=1), L


def extract_codebook(vq_net: nn.Module) -> torch.Tensor:
    """
    Robustly extract codebook weights from common VQ implementations.

    Return:
        codebook: [K, D_vq]
    """
    if hasattr(vq_net, "quantizer"):
        q = vq_net.quantizer

        if hasattr(q, "codebook"):
            cb = q.codebook

            if isinstance(cb, nn.Parameter):
                return cb.detach().cpu()

            if torch.is_tensor(cb):
                return cb.detach().cpu()

            if hasattr(cb, "weight"):
                return cb.weight.detach().cpu()

        if hasattr(q, "embedding"):
            emb = q.embedding

            if isinstance(emb, nn.Embedding):
                return emb.weight.detach().cpu()

            if torch.is_tensor(emb):
                return emb.detach().cpu()

    raise AttributeError("Cannot extract codebook from vq_net.quantizer.")


def load_codebook_from_path_or_vq(
    vq_net: nn.Module,
    ckpt_dir: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Load best_codebook.pth if it exists; otherwise extract from vq_net.

    Supported checkpoint formats:
        Tensor
        {"codebook": Tensor}
        {"weight": Tensor}
    """
    cb_path = os.path.join(ckpt_dir, "best_codebook.pth")

    if os.path.exists(cb_path):
        codebook = torch.load(cb_path, map_location="cpu")

        if isinstance(codebook, dict):
            if "codebook" in codebook:
                codebook = codebook["codebook"]
            elif "weight" in codebook:
                codebook = codebook["weight"]
            else:
                raise KeyError(
                    f"Unknown codebook checkpoint format in {cb_path}. "
                    f"Expected keys: 'codebook' or 'weight'. Got: {list(codebook.keys())}"
                )
    else:
        codebook = extract_codebook(vq_net)

    if not torch.is_tensor(codebook):
        raise TypeError(f"Loaded codebook is not a tensor: {type(codebook)}")

    if codebook.dim() != 2:
        raise ValueError(f"Codebook should be [K, D], got {tuple(codebook.shape)}")

    return codebook.float().to(device)


def load_adapter_flexible(
    module: nn.Module,
    checkpoint: Dict[str, Any],
    key: str,
) -> bool:
    """
    Load a submodule from flexible checkpoint formats.

    Supported:
        checkpoint[key]
        checkpoint['state_dict'] with prefix key + '.'
        checkpoint with prefix key + '.'
    """
    if key in checkpoint and isinstance(checkpoint[key], dict):
        module.load_state_dict(checkpoint[key], strict=True)
        return True

    state = checkpoint.get("state_dict", checkpoint)

    prefix = key + "."
    sub_state = {}

    for k, v in state.items():
        if k.startswith(prefix):
            sub_state[k[len(prefix):]] = v

    if len(sub_state) > 0:
        module.load_state_dict(sub_state, strict=True)
        return True

    return False


def _safe_json_load(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Primitive LLM Teacher
# ============================================================
class PrimitiveLLMTeacherCore(nn.Module):
    """
    Primitive-aware LLM teacher for wearable HAR.

    It supports three roles:

    1) Masked primitive recovery:
        visible primitive ids -> missing primitive distribution

    2) Activity classification:
        primitive sequence -> activity logits

    3) Explanation generation:
        primitive sequence/profile -> textual explanation

    Main pathway:
        primitive ids
        -> VQ codebook embeddings
        -> projector into LLM embedding space
        -> optional primitive semantic embeddings
        -> position embeddings
        -> frozen causal LLM
        -> query hidden states
        -> primitive recovery head and activity classification head

    Important:
        - The LLM backbone is frozen by default.
        - During teacher alignment, call freeze_llm_only().
        - During distillation/inference, call freeze_all().
    """

    def __init__(
        self,
        llm_path: str,
        codebook_weights: torch.Tensor,
        mask_token_id: int,
        max_patches: int,
        device: torch.device,
        temperature: float = 1.0,
        torch_dtype: Optional[torch.dtype] = None,
        system_prompt: str = (
            "Analyze wearable IMU motion primitives. "
            "Use primitive semantic profiles to recover masked primitives and classify the activity."
        ),
        num_classes: Optional[int] = None,
        label_names: Optional[List[str]] = None,
        primitive_profile_path: Optional[str] = None,
        primitive_descriptions: Optional[Union[Dict[int, str], Dict[str, str]]] = None,
        use_semantic_primitive: bool = True,
        semantic_weight: float = 0.5,
        enable_explanation: bool = True,
    ):
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required for PrimitiveLLMTeacherCore.")

        self.device = device
        self.mask_token_id = int(mask_token_id)
        self.max_patches = int(max_patches)
        self.temperature = float(temperature)
        self.system_prompt = str(system_prompt)
        self.use_semantic_primitive = bool(use_semantic_primitive)
        self.semantic_weight = float(semantic_weight)
        self.enable_explanation = bool(enable_explanation)

        if self.max_patches <= 0:
            raise ValueError(f"max_patches must be positive, got {self.max_patches}")

        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")

        if torch_dtype is None:
            torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

        print(f"[PrimitiveLLMTeacherCore] Loading LLM from: {llm_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            use_fast=False,
            trust_remote_code=True,
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device).eval()

        if hasattr(self.llm, "config"):
            self.llm.config.use_cache = False

        for p in self.llm.parameters():
            p.requires_grad = False

        self.llm_dim = int(self.llm.config.hidden_size)

        # ========================================================
        # Codebook
        # ========================================================
        codebook_weights = codebook_weights.float()

        if codebook_weights.dim() != 2:
            raise ValueError(
                f"codebook_weights must be [K, D], got {tuple(codebook_weights.shape)}"
            )

        self.register_buffer("codebook", codebook_weights.to(device))

        self.num_vq_codes = int(self.codebook.shape[0])
        self.vq_dim = int(self.codebook.shape[1])

        if self.num_vq_codes <= 0 or self.vq_dim <= 0:
            raise ValueError(f"Invalid codebook shape: {tuple(self.codebook.shape)}")

        # ========================================================
        # Label names and class count
        # ========================================================
        if label_names is not None:
            self.label_names = [str(x) for x in label_names]
            inferred_num_classes = len(self.label_names)
        else:
            self.label_names = None
            inferred_num_classes = None

        if num_classes is None:
            num_classes = inferred_num_classes

        self.num_classes = None if num_classes is None else int(num_classes)

        if self.num_classes is not None and self.num_classes <= 0:
            raise ValueError(f"num_classes should be positive, got {self.num_classes}")

        if self.label_names is not None and self.num_classes != len(self.label_names):
            raise ValueError(
                f"num_classes={self.num_classes}, but len(label_names)={len(self.label_names)}"
            )

        # ========================================================
        # Trainable adapters
        # ========================================================
        self.projector = nn.Sequential(
            nn.LayerNorm(self.vq_dim),
            nn.Linear(self.vq_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim),
        ).to(device)

        self.mask_embed_llama = nn.Parameter(
            torch.randn(1, 1, self.llm_dim, device=device) * 0.02
        )

        self.primitive_pos_embed = nn.Parameter(
            torch.randn(1, self.max_patches, self.llm_dim, device=device) * 0.02
        )

        self.query_embed = nn.Parameter(
            torch.randn(1, self.max_patches, self.llm_dim, device=device) * 0.02
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(self.llm_dim),
            nn.Linear(self.llm_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.num_vq_codes),
        ).to(device)

        if self.num_classes is not None:
            self.activity_head = nn.Sequential(
                nn.LayerNorm(self.llm_dim),
                nn.Linear(self.llm_dim, self.llm_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.llm_dim, self.num_classes),
            ).to(device)
        else:
            self.activity_head = None

        # ========================================================
        # Prompt embedding
        # ========================================================
        prompt_ids = self.tokenizer(
            self.system_prompt,
            return_tensors="pt",
        ).input_ids.to(device)

        self.register_buffer("prompt_input_ids", prompt_ids)

        with torch.no_grad():
            prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)

        self.register_buffer("prompt_embeds", prompt_embeds)

        # ========================================================
        # Primitive semantic embeddings
        # ========================================================
        profile = _safe_json_load(primitive_profile_path)

        if primitive_descriptions is None and profile is not None:
            primitive_descriptions = self._extract_descriptions_from_profile(profile)

        self.primitive_descriptions = self._normalize_primitive_descriptions(
            primitive_descriptions
        )

        semantic_embed = self._build_primitive_semantic_embeddings(
            self.primitive_descriptions
        )

        self.register_buffer("primitive_semantic_embed", semantic_embed)

        # ========================================================
        # Init
        # ========================================================
        self._init_trainable_modules()

    # ========================================================
    # Initialization
    # ========================================================
    def _init_trainable_modules(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ========================================================
    # Primitive description helpers
    # ========================================================
    def _extract_descriptions_from_profile(
            self,
            profile: Dict[str, Any],
    ) -> Dict[int, str]:
        out: Dict[int, str] = {}

        for k, v in profile.items():
            try:
                idx = int(k)
            except Exception:
                continue

            if isinstance(v, str):
                out[idx] = v

            elif isinstance(v, dict):
                desc = None

                desc_parts = []

                if "description" in v and v["description"] is not None:
                    desc_parts.append(str(v["description"]))

                if "case_study" in v and isinstance(v["case_study"], dict):
                    if "summary" in v["case_study"] and v["case_study"]["summary"] is not None:
                        desc_parts.append(str(v["case_study"]["summary"]))

                if len(desc_parts) > 0:
                    desc = " ".join(desc_parts)

                if desc is None:
                    stats = v.get("stats", {})
                    label_dist = v.get("label_distribution", [])

                    parts = [f"motion primitive {idx}"]

                    if isinstance(stats, dict):
                        if "dominant_channel_name" in stats:
                            parts.append(f"dominant channel: {stats['dominant_channel_name']}")
                        if "energy_mean" in stats:
                            parts.append(f"energy: {float(stats['energy_mean']):.4f}")
                        if "periodicity_mean" in stats:
                            parts.append(f"periodicity: {float(stats['periodicity_mean']):.4f}")

                    if isinstance(label_dist, list) and len(label_dist) > 0:
                        top = label_dist[0]
                        parts.append(
                            f"associated with {top.get('label_name', 'unknown')} "
                            f"({float(top.get('ratio', 0.0)):.2f})"
                        )

                    desc = "; ".join(parts)

                out[idx] = desc

            else:
                out[idx] = f"motion primitive {idx}"

        return out
    def _normalize_primitive_descriptions(
        self,
        primitive_descriptions: Optional[Union[Dict[int, str], Dict[str, str]]],
    ) -> Dict[int, str]:
        out: Dict[int, str] = {}

        if primitive_descriptions is not None:
            for k, v in primitive_descriptions.items():
                try:
                    idx = int(k)
                except Exception:
                    continue

                if 0 <= idx < self.num_vq_codes:
                    out[idx] = str(v)

        for idx in range(self.num_vq_codes):
            if idx not in out:
                out[idx] = f"motion primitive {idx}"

        return out

    @torch.no_grad()
    def _build_primitive_semantic_embeddings(
        self,
        primitive_descriptions: Dict[int, str],
    ) -> torch.Tensor:
        """
        Build frozen semantic embeddings for each primitive description.

        Return:
            semantic_embed: [K, D_llm]
        """
        embeds = []

        input_embedding = self.llm.get_input_embeddings()

        for idx in range(self.num_vq_codes):
            desc = primitive_descriptions.get(idx, f"motion primitive {idx}")
            text = (
                f"Motion primitive {idx}. "
                f"Semantic profile: {desc}. "
                f"This primitive may indicate local IMU intensity, periodicity, dominant sensor channel, "
                f"and activity-related motion evidence."
            )

            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=48,
            )

            input_ids = encoded.input_ids.to(self.device)
            attn_mask = encoded.attention_mask.to(self.device).float()

            token_embeds = input_embedding(input_ids)  # [1, T, D]

            denom = attn_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (token_embeds * attn_mask.unsqueeze(-1)).sum(dim=1) / denom

            embeds.append(pooled.squeeze(0).float())

        semantic_embed = torch.stack(embeds, dim=0).to(self.device)  # [K, D]
        return semantic_embed

    # ========================================================
    # Freezing controls
    # ========================================================
    def freeze_llm_only(self):
        """
        Alignment stage:
            freeze LLM;
            train projector / mask / position / query / output_head / activity_head.
        """
        for p in self.llm.parameters():
            p.requires_grad = False

        for p in self.projector.parameters():
            p.requires_grad = True

        for p in self.output_head.parameters():
            p.requires_grad = True

        if self.activity_head is not None:
            for p in self.activity_head.parameters():
                p.requires_grad = True

        self.mask_embed_llama.requires_grad_(True)
        self.primitive_pos_embed.requires_grad_(True)
        self.query_embed.requires_grad_(True)

        # primitive_semantic_embed is a frozen buffer.
        self.train()

    def freeze_all(self):
        """
        Distillation / inference stage:
            freeze all teacher parameters.
        """
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    # ========================================================
    # Forward
    # ========================================================
    def forward(
            self,
            masked_ids: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            target_ids: Optional[torch.Tensor] = None,
            valid_mask: Optional[torch.Tensor] = None,
            return_loss: bool = False,
            primitive_loss_mask: Optional[torch.Tensor] = None,
            lambda_activity: float = 1.0,
            generate_explanation: bool = False,
            max_new_tokens: int = 96,
    ) -> Union[TeacherOut, Tuple[torch.Tensor, TeacherOut, Dict[str, float]]]:
        """
        Args:
            masked_ids:
                [B, P], values in [0, K-1] or mask_token_id.

            labels:
                Optional activity labels, [B].

            target_ids:
                Original unmasked primitive ids, [B, P].
                Required when return_loss=True and primitive_loss_mask has True values.

            valid_mask:
                Optional [B, P], True means valid primitive position.
                Used for pooling and metric computation.

            return_loss:
                If True, return (loss, TeacherOut, metrics).

            primitive_loss_mask:
                Optional bool mask [B, P]. True means positions supervised by primitive recovery.

            lambda_activity:
                Weight for activity classification loss.

            generate_explanation:
                If True, generate textual explanations. Slow; only use for case study.
        """
        masked_ids = masked_ids.to(self.device).long()

        if masked_ids.dim() != 2:
            raise ValueError(f"masked_ids should be [B, P], got {tuple(masked_ids.shape)}")

        B, P = masked_ids.shape

        if P != self.max_patches:
            raise ValueError(
                f"Primitive length mismatch: input P={P}, teacher max_patches={self.max_patches}."
            )

        if valid_mask is None:
            valid_mask = torch.ones((B, P), device=self.device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(self.device).bool()

        is_mask = masked_ids == self.mask_token_id

        safe_ids = masked_ids.clone()
        safe_ids[is_mask] = 0
        safe_ids = safe_ids.clamp(0, self.num_vq_codes - 1)

        # ------------------------------------------------------------
        # 1. VQ codebook embedding
        # ------------------------------------------------------------
        vq_embeds = F.embedding(safe_ids, self.codebook)  # [B, P, D_vq]
        sensor_embeds = self.projector(vq_embeds)  # [B, P, D_llm]

        # ------------------------------------------------------------
        # 2. Semantic primitive embedding
        # ------------------------------------------------------------
        if self.use_semantic_primitive and self.primitive_semantic_embed is not None:
            semantic_embeds = F.embedding(
                safe_ids,
                self.primitive_semantic_embed,
            ).to(sensor_embeds.dtype)

            sensor_embeds = sensor_embeds + self.semantic_weight * semantic_embeds

        # ------------------------------------------------------------
        # 3. Replace masked positions
        # ------------------------------------------------------------
        mask_embeds = self.mask_embed_llama.expand(B, P, -1)

        sensor_embeds = torch.where(
            is_mask.unsqueeze(-1),
            mask_embeds,
            sensor_embeds,
        )

        # Invalid padding positions should not carry real primitive semantics.
        sensor_embeds = torch.where(
            valid_mask.unsqueeze(-1),
            sensor_embeds,
            mask_embeds,
        )

        # ------------------------------------------------------------
        # 4. Position embedding
        # ------------------------------------------------------------
        sensor_embeds = sensor_embeds + self.primitive_pos_embed[:, :P, :]

        # ------------------------------------------------------------
        # 5. Prompt + primitive sequence + query tokens
        # ------------------------------------------------------------
        prompt_embeds = self.prompt_embeds.expand(B, -1, -1)
        query_embeds = self.query_embed[:, :P, :].expand(B, -1, -1)

        inputs_embeds = torch.cat(
            [prompt_embeds, sensor_embeds, query_embeds],
            dim=1,
        ).to(self.llm.dtype)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            use_cache=False,
        )

        last_hidden = outputs.hidden_states[-1]

        query_start = prompt_embeds.shape[1] + P
        query_end = query_start + P

        query_hidden = last_hidden[:, query_start:query_end, :]  # [B, P, D]

        # ------------------------------------------------------------
        # 6. Primitive recovery
        # ------------------------------------------------------------
        primitive_logits = self.output_head(query_hidden.float())  # [B, P, K]
        primitive_probs = F.softmax(primitive_logits / self.temperature, dim=-1)

        # ------------------------------------------------------------
        # 7. Activity classification
        # ------------------------------------------------------------
        activity_logits = None
        activity_probs = None

        if self.activity_head is not None:
            valid_float = valid_mask.float().unsqueeze(-1)  # [B, P, 1]
            denom = valid_float.sum(dim=1).clamp_min(1.0)
            global_hidden = (query_hidden.float() * valid_float).sum(dim=1) / denom

            activity_logits = self.activity_head(global_hidden)
            activity_probs = F.softmax(activity_logits, dim=-1)

        explanation_texts = None

        if generate_explanation:
            if not self.enable_explanation:
                raise RuntimeError("Explanation generation is disabled.")

            explanation_texts = self.generate_explanations(
                masked_ids=masked_ids,
                activity_probs=activity_probs,
                max_new_tokens=max_new_tokens,
            )

        out = TeacherOut(
            logits=primitive_logits,
            probs=primitive_probs,
            activity_logits=activity_logits,
            activity_probs=activity_probs,
            explanation_texts=explanation_texts,
        )

        if not return_loss:
            return out

        # ------------------------------------------------------------
        # 8. Loss
        # ------------------------------------------------------------
        if primitive_loss_mask is None:
            primitive_loss_mask = is_mask & valid_mask
        else:
            primitive_loss_mask = primitive_loss_mask.to(self.device).bool() & valid_mask

        if target_ids is None:
            target_ids = masked_ids.clone()
        else:
            target_ids = target_ids.to(self.device).long()

        if target_ids.shape != masked_ids.shape:
            raise ValueError(
                f"target_ids shape {tuple(target_ids.shape)} does not match masked_ids {tuple(masked_ids.shape)}"
            )

        metrics: Dict[str, float] = {}

        # primitive recovery loss
        if primitive_loss_mask.sum() > 0:
            pred_prim = primitive_logits[primitive_loss_mask]
            target_prim = target_ids[primitive_loss_mask].clamp(0, self.num_vq_codes - 1)

            loss_primitive = F.cross_entropy(pred_prim, target_prim)

            with torch.no_grad():
                prim_pred = pred_prim.argmax(dim=-1)
                prim_acc = (prim_pred == target_prim).float().mean()
                prim_prob = F.softmax(pred_prim, dim=-1)
                prim_conf = prim_prob.max(dim=-1).values.mean()
                prim_entropy = -(prim_prob * (prim_prob + 1e-8).log()).sum(dim=-1).mean()
                prim_entropy_norm = prim_entropy / torch.log(
                    torch.tensor(float(self.num_vq_codes), device=self.device)
                )
        else:
            loss_primitive = torch.tensor(0.0, device=self.device, requires_grad=True)
            prim_acc = torch.tensor(0.0, device=self.device)
            prim_conf = torch.tensor(0.0, device=self.device)
            prim_entropy = torch.tensor(0.0, device=self.device)
            prim_entropy_norm = torch.tensor(0.0, device=self.device)

        # activity classification loss
        loss_activity = torch.tensor(0.0, device=self.device)

        if labels is not None:
            if self.activity_head is None or activity_logits is None:
                raise RuntimeError("labels are provided but activity_head is not enabled.")

            labels = labels.to(self.device).long()
            loss_activity = F.cross_entropy(activity_logits, labels)

            with torch.no_grad():
                cls_pred = activity_logits.argmax(dim=-1)
                cls_acc = (cls_pred == labels).float().mean()
        else:
            cls_acc = torch.tensor(0.0, device=self.device)

        loss_total = loss_primitive + float(lambda_activity) * loss_activity

        metrics["loss_total"] = float(loss_total.detach().item())
        metrics["loss_primitive"] = float(loss_primitive.detach().item())
        metrics["loss_activity"] = float(loss_activity.detach().item())
        metrics["primitive_acc"] = float(prim_acc.detach().item())
        metrics["primitive_conf"] = float(prim_conf.detach().item())
        metrics["primitive_entropy"] = float(prim_entropy.detach().item())
        metrics["primitive_entropy_norm"] = float(prim_entropy_norm.detach().item())
        metrics["activity_acc"] = float(cls_acc.detach().item())

        return loss_total, out, metrics

    # ========================================================
    # Explanation generation
    # ========================================================
    @torch.no_grad()
    def generate_explanations(
        self,
        masked_ids: torch.Tensor,
        activity_probs: Optional[torch.Tensor] = None,
        max_new_tokens: int = 96,
    ) -> List[str]:
        """
        Generate human-readable explanations from primitive descriptions.

        This function uses textual prompts and the LLM generation API.
        It is intended for case studies / qualitative analysis, not for
        inner-loop training.
        """
        if not self.enable_explanation:
            raise RuntimeError("Explanation generation is disabled.")

        masked_ids = masked_ids.to(self.device).long()

        B, P = masked_ids.shape
        texts: List[str] = []

        for b in range(B):
            ids = masked_ids[b].detach().cpu().tolist()

            primitive_lines = []

            for t, idx in enumerate(ids):
                if idx == self.mask_token_id:
                    desc = "[MASKED primitive]"
                elif 0 <= idx < self.num_vq_codes:
                    desc = self.primitive_descriptions.get(
                        int(idx),
                        f"motion primitive {idx}",
                    )
                else:
                    desc = "[INVALID primitive]"

                primitive_lines.append(f"{t + 1}. {desc}")

            candidate_text = ""

            if self.label_names is not None:
                candidate_text = "Candidate activities: " + ", ".join(self.label_names) + ".\n"

            pred_text = ""

            if activity_probs is not None and self.label_names is not None:
                probs_b = activity_probs[b].detach().cpu()
                topk = min(3, probs_b.numel())
                vals, inds = torch.topk(probs_b, k=topk)

                pairs = []
                for v, i in zip(vals.tolist(), inds.tolist()):
                    pairs.append(f"{self.label_names[i]} ({v:.3f})")

                pred_text = "Teacher top predictions: " + ", ".join(pairs) + ".\n"

            prompt = (
                    "You are writing a concise case-study explanation for a wearable human activity recognition model.\n"
                    "The input sequence is represented by interpretable motion primitives. "
                    "Each primitive profile describes its activity association, motion intensity, temporal variation, "
                    "periodicity, spectral structure, dominant sensor channel, and transition context.\n\n"
                    f"{candidate_text}"
                    f"{pred_text}"
                    "Observed primitive sequence:\n"
                    + "\n".join(primitive_lines)
                    + "\n\n"
                      "Write the explanation in the following style:\n"
                      "1. Identify the most likely activity.\n"
                      "2. Mention the key primitives that support this decision.\n"
                      "3. Explain the evidence using motion intensity, periodicity, posture/stability, dominant channel, "
                      "and repeated primitive patterns.\n"
                      "4. Keep the explanation concise and suitable for an academic paper case study.\n\n"
                      "Explanation:"
            )

            encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            generated = self.llm.generate(
                **encoded,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            decoded = self.tokenizer.decode(
                generated[0][encoded.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            texts.append(decoded.strip())

        return texts

    # ========================================================
    # Adapter save/load
    # ========================================================
    def save_adapter(self, path: str):
        """
        Save only lightweight adapters, not the frozen LLM.
        """
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        payload = {
            "projector": self.projector.state_dict(),
            "output_head": self.output_head.state_dict(),
            "mask_embed_llama": self.mask_embed_llama.detach().cpu(),
            "primitive_pos_embed": self.primitive_pos_embed.detach().cpu(),
            "query_embed": self.query_embed.detach().cpu(),
            "primitive_descriptions": self.primitive_descriptions,
            "meta": {
                "num_vq_codes": int(self.num_vq_codes),
                "vq_dim": int(self.vq_dim),
                "llm_dim": int(self.llm_dim),
                "max_patches": int(self.max_patches),
                "system_prompt": str(self.system_prompt),
                "num_classes": None if self.num_classes is None else int(self.num_classes),
                "label_names": self.label_names,
                "use_semantic_primitive": bool(self.use_semantic_primitive),
                "semantic_weight": float(self.semantic_weight),
            },
        }

        if self.activity_head is not None:
            payload["activity_head"] = self.activity_head.state_dict()

        torch.save(payload, path)

        print(f"[PrimitiveLLMTeacherCore] Adapter saved to: {path}")

    def load_adapter(self, path: str, strict: bool = True):
        """
        Load aligned lightweight adapters.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        ckpt = torch.load(path, map_location=self.device)

        meta = ckpt.get("meta", {})

        if "max_patches" in meta and int(meta["max_patches"]) != self.max_patches:
            raise ValueError(
                f"Adapter max_patches mismatch: ckpt={meta['max_patches']}, "
                f"current={self.max_patches}. Please use the matching alignment checkpoint."
            )

        if "num_vq_codes" in meta and int(meta["num_vq_codes"]) != self.num_vq_codes:
            raise ValueError(
                f"Adapter num_vq_codes mismatch: ckpt={meta['num_vq_codes']}, "
                f"current={self.num_vq_codes}. Please check VQ-VAE and alignment checkpoint."
            )

        if "vq_dim" in meta and int(meta["vq_dim"]) != self.vq_dim:
            raise ValueError(
                f"Adapter vq_dim mismatch: ckpt={meta['vq_dim']}, current={self.vq_dim}."
            )

        ok_projector = load_adapter_flexible(self.projector, ckpt, "projector")
        ok_output = load_adapter_flexible(self.output_head, ckpt, "output_head")

        ok_activity = True

        if self.activity_head is not None:
            if "activity_head" in ckpt:
                self.activity_head.load_state_dict(ckpt["activity_head"], strict=True)
                ok_activity = True
            else:
                ok_activity = False

        required_tensor_keys = [
            "mask_embed_llama",
            "primitive_pos_embed",
            "query_embed",
        ]

        missing_tensor_keys = [k for k in required_tensor_keys if k not in ckpt]

        if strict and len(missing_tensor_keys) > 0:
            raise KeyError(f"Missing adapter tensor keys: {missing_tensor_keys}")

        with torch.no_grad():
            if "mask_embed_llama" in ckpt:
                self.mask_embed_llama.copy_(ckpt["mask_embed_llama"].to(self.device))

            if "primitive_pos_embed" in ckpt:
                pos = ckpt["primitive_pos_embed"].to(self.device)

                if pos.shape != self.primitive_pos_embed.shape:
                    raise ValueError(
                        f"primitive_pos_embed shape mismatch: ckpt={tuple(pos.shape)}, "
                        f"current={tuple(self.primitive_pos_embed.shape)}"
                    )

                self.primitive_pos_embed.copy_(pos)

            if "query_embed" in ckpt:
                query = ckpt["query_embed"].to(self.device)

                if query.shape != self.query_embed.shape:
                    raise ValueError(
                        f"query_embed shape mismatch: ckpt={tuple(query.shape)}, "
                        f"current={tuple(self.query_embed.shape)}"
                    )

                self.query_embed.copy_(query)

        if "primitive_descriptions" in ckpt:
            self.primitive_descriptions = self._normalize_primitive_descriptions(
                ckpt["primitive_descriptions"]
            )

            semantic_embed = self._build_primitive_semantic_embeddings(
                self.primitive_descriptions
            )

            self.primitive_semantic_embed.copy_(semantic_embed)

        if strict and not (ok_projector and ok_output and ok_activity):
            raise RuntimeError(
                f"Failed to load adapter completely: "
                f"projector={ok_projector}, output_head={ok_output}, activity_head={ok_activity}"
            )

        return {
            "projector": ok_projector,
            "output_head": ok_output,
            "activity_head": ok_activity,
            "mask_embed": "mask_embed_llama" in ckpt,
            "pos_embed": "primitive_pos_embed" in ckpt,
            "query_embed": "query_embed" in ckpt,
            "has_primitive_descriptions": "primitive_descriptions" in ckpt,
        }