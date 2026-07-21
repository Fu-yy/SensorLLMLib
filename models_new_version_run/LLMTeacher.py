# models_new_version_run/primitive_llm_teacher.py
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

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


@dataclass
class TeacherOut:
    logits: torch.Tensor  # [B, P, K]
    probs: torch.Tensor   # [B, P, K]


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
    Return shape: [K, D_vq].
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


def load_codebook_from_path_or_vq(vq_net: nn.Module, ckpt_dir: str, device: torch.device) -> torch.Tensor:
    """
    Load best_codebook.pth if it exists; otherwise extract from vq_net.
    Handles both tensor and dict checkpoint formats.
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

    return codebook.float().to(device)


def load_adapter_flexible(module: nn.Module, checkpoint: Dict[str, Any], key: str) -> bool:
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


class PrimitiveLLMTeacherCore(nn.Module):
    """
    Shared LLM teacher core for both:
        1) teacher alignment stage
        2) main student distillation stage

    Structure:
        VQ primitive ids
        -> codebook embedding
        -> projector into LLM embedding space
        -> masked primitive sequence + primitive position embeddings
        -> append query tokens
        -> frozen causal LLM
        -> query hidden states
        -> output_head predicts primitive distribution

    Important:
        - LLM is always frozen.
        - During alignment, only adapters are trainable.
        - During student distillation, everything is frozen.
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
        system_prompt: str = "Recover masked sensor primitives:",
    ):
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required for PrimitiveLLMTeacherCore.")

        self.device = device
        self.mask_token_id = int(mask_token_id)
        self.max_patches = int(max_patches)
        self.temperature = float(temperature)
        self.system_prompt = str(system_prompt)

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
        # ============================================================
        # Fix missing chat template for local Llama-Instruct folders
        # ============================================================
        self._ensure_chat_template(llm_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device).eval()

        # Disable KV cache for training-style forward with inputs_embeds.
        if hasattr(self.llm, "config"):
            self.llm.config.use_cache = False

        for p in self.llm.parameters():
            p.requires_grad = False

        self.llm_dim = int(self.llm.config.hidden_size)

        # -------------------------
        # Codebook
        # -------------------------
        codebook_weights = codebook_weights.float()
        if codebook_weights.dim() != 2:
            raise ValueError(f"codebook_weights must be [K, D], got {tuple(codebook_weights.shape)}")

        self.register_buffer("codebook", codebook_weights.to(device))

        self.num_vq_codes = int(self.codebook.shape[0])
        self.vq_dim = int(self.codebook.shape[1])

        if self.num_vq_codes <= 0 or self.vq_dim <= 0:
            raise ValueError(f"Invalid codebook shape: {tuple(self.codebook.shape)}")

        # -------------------------
        # Trainable adapters
        # Must remain identical across alignment and distillation.
        # -------------------------
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

        # -------------------------
        # Prompt embedding
        # -------------------------
        prompt_ids = self.tokenizer(
            self.system_prompt,
            return_tensors="pt",
        ).input_ids.to(device)

        self.register_buffer("prompt_input_ids", prompt_ids)

        with torch.no_grad():
            prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)

        self.register_buffer("prompt_embeds", prompt_embeds)

    def _ensure_chat_template(self, llm_path: str):
        """
        Ensure chat template is available.

        Some local Llama-Instruct folders may miss tokenizer.chat_template.
        If chat_template is missing, apply a Llama-3 style template manually.
        """

        current_template = getattr(self.tokenizer, "chat_template", None)

        if current_template is not None and str(current_template).strip() != "":
            print("[PrimitivePromptLLM] tokenizer.chat_template exists.")
            return

        llm_path_lower = str(llm_path).lower()

        # Llama-3 / Llama-3.1 / Llama-3.2 instruct template
        if "llama" in llm_path_lower and "instruct" in llm_path_lower:
            print(
                "[PrimitivePromptLLM][warn] tokenizer.chat_template is missing. Use manual Llama-3 instruct template.")

            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}"
                "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}"
                "{{ message['content'] | trim }}"
                "{{ '<|eot_id|>' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
                "{% endif %}"
            )

            return

        # Generic fallback for Qwen / other instruct models should usually have their own template.
        # If not, we keep None and use plain text fallback in _format_generation_text().
        print("[PrimitivePromptLLM][warn] tokenizer.chat_template is missing and no known manual template is applied.")
    def freeze_llm_only(self):
        """
        Alignment stage:
            freeze LLM, train projector / mask / position / query / output_head.
        """
        for p in self.llm.parameters():
            p.requires_grad = False

        for p in self.projector.parameters():
            p.requires_grad = True

        for p in self.output_head.parameters():
            p.requires_grad = True

        self.mask_embed_llama.requires_grad_(True)
        self.primitive_pos_embed.requires_grad_(True)
        self.query_embed.requires_grad_(True)

        self.train()

    def freeze_all(self):
        """
        Distillation stage:
            freeze all teacher parameters.
        """
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, masked_ids: torch.Tensor) -> TeacherOut:
        """
        Args:
            masked_ids: [B, P], values in [0, K-1] or mask_token_id.

        Returns:
            TeacherOut:
                logits: [B, P, K]
                probs : [B, P, K], temperature-scaled
        """
        masked_ids = masked_ids.to(self.device).long()

        B, P = masked_ids.shape

        if P != self.max_patches:
            raise ValueError(
                f"Primitive length mismatch: input P={P}, teacher max_patches={self.max_patches}. "
                "Alignment stage and student pretraining must use the same seq_len and VQ stride."
            )

        is_mask = masked_ids == self.mask_token_id

        safe_ids = masked_ids.clone()
        safe_ids[is_mask] = 0
        safe_ids = safe_ids.clamp(0, self.num_vq_codes - 1)

        # [B, P, D_vq]
        vq_embeds = F.embedding(safe_ids, self.codebook)

        # [B, P, D_llm]
        sensor_embeds = self.projector(vq_embeds)

        # Replace masked primitive embeddings.
        mask_embeds = self.mask_embed_llama.expand(B, P, -1)
        sensor_embeds = torch.where(
            is_mask.unsqueeze(-1),
            mask_embeds,
            sensor_embeds,
        )

        # Add primitive position embeddings.
        sensor_embeds = sensor_embeds + self.primitive_pos_embed[:, :P, :]

        # Prompt + masked primitive sequence + query tokens.
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

        # [B, P, D_llm]
        query_hidden = last_hidden[:, query_start:query_end, :]

        # [B, P, K]
        logits = self.output_head(query_hidden.float())

        probs = F.softmax(logits / self.temperature, dim=-1)

        return TeacherOut(logits=logits, probs=probs)

    def save_adapter(self, path: str):
        """
        Save only the lightweight adapters, not the frozen LLM.
        """
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(
            {
                "projector": self.projector.state_dict(),
                "output_head": self.output_head.state_dict(),
                "mask_embed_llama": self.mask_embed_llama.detach().cpu(),
                "primitive_pos_embed": self.primitive_pos_embed.detach().cpu(),
                "query_embed": self.query_embed.detach().cpu(),
                "meta": {
                    "num_vq_codes": int(self.num_vq_codes),
                    "vq_dim": int(self.vq_dim),
                    "llm_dim": int(self.llm_dim),
                    "max_patches": int(self.max_patches),
                    "system_prompt": str(self.system_prompt),
                },
            },
            path,
        )

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

        ok_projector = load_adapter_flexible(self.projector, ckpt, "projector")
        ok_output = load_adapter_flexible(self.output_head, ckpt, "output_head")

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

        if strict and not (ok_projector and ok_output):
            raise RuntimeError(
                f"Failed to load adapter completely: "
                f"projector={ok_projector}, output_head={ok_output}"
            )

        return {
            "projector": ok_projector,
            "output_head": ok_output,
            "mask_embed": "mask_embed_llama" in ckpt,
            "pos_embed": "primitive_pos_embed" in ckpt,
            "query_embed": "query_embed" in ckpt,
        }