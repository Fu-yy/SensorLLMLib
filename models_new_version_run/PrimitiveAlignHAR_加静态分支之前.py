import os
import json
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml
except Exception:
    yaml = None

from transformers import AutoTokenizer, AutoModelForCausalLM
from models_new_version_run.VQ_VAE import IMU_VQ_Model


def pad_to_multiple(x: torch.Tensor, multiple: int, pad_value: float = 0.0):
    B, L, C = x.shape
    L_pad = ((L + multiple - 1) // multiple) * multiple
    if L_pad == L:
        return x, L
    pad_len = L_pad - L
    pad = x.new_full((B, pad_len, C), pad_value)
    return torch.cat([x, pad], dim=1), L


def get_label_names_from_cfg(ds_cfg: Dict[str, Any], num_class: int):
    if "label_names" in ds_cfg and ds_cfg["label_names"] is not None:
        return [str(x) for x in ds_cfg["label_names"]]

    if "id2label" in ds_cfg and ds_cfg["id2label"] is not None:
        id2label = {int(k): str(v) for k, v in ds_cfg["id2label"].items()}
        return [id2label[i] for i in range(num_class)]

    return [f"class_{i}" for i in range(num_class)]

def get_channel_names_from_cfg(ds_cfg: Dict[str, Any], channel_num: int):
    if "channel_names" in ds_cfg and ds_cfg["channel_names"] is not None:
        names = [str(x) for x in ds_cfg["channel_names"]]
        if len(names) == channel_num:
            return names

    inferred = []
    for k in ds_cfg.keys():
        if str(k).startswith("default_") and str(k).endswith("_start_token"):
            name = str(k).replace("default_", "").replace("_start_token", "")
            inferred.append(name)

    unique = []
    for name in inferred:
        if name not in unique:
            unique.append(name)

    if len(unique) == channel_num:
        return unique

    return [f"channel_{i}" for i in range(channel_num)]
def load_profile(profile_path: str):
    if profile_path is None or not os.path.exists(profile_path):
        raise FileNotFoundError(f"primitive profile not found: {profile_path}")
    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_codebook(vq_net: nn.Module) -> torch.Tensor:
    q = vq_net.quantizer

    if hasattr(q, "codebook"):
        cb = q.codebook
        if torch.is_tensor(cb):
            return cb.detach()
        if hasattr(cb, "weight"):
            return cb.weight.detach()

    if hasattr(q, "embedding"):
        emb = q.embedding
        if isinstance(emb, nn.Embedding):
            return emb.weight.detach()
        if torch.is_tensor(emb):
            return emb.detach()

    raise AttributeError("Cannot extract VQ codebook.")


class AlignmentModel(nn.Module):
    """
    Trainable primitive-language alignment model for HAR.

    Frozen:
        - VQ-VAE
        - LLM/text encoder used to encode primitive profile

    Trainable:
        - VQ embedding projector
        - profile semantic projector
        - Transformer encoder
        - activity classification head
        - masked primitive recovery head
    """

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device = args.device

        self.dataset_key = str(
            getattr(args, "dataset_key", getattr(args, "data", "mhealth"))
        ).lower()

        if hasattr(args, "ds_cfg") and isinstance(args.ds_cfg, dict):
            self.ds_cfg = args.ds_cfg
        else:
            self.ds_cfg = self._load_ds_cfg_from_yaml(args)

        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 15)))
        self.num_class = int(self.ds_cfg.get("num_labels", getattr(args, "num_class", 12)))
        self.seq_len_orig = int(getattr(args, "seq_len", 200))

        self.label_names = get_label_names_from_cfg(self.ds_cfg, self.num_class)
        self.channel_names = get_channel_names_from_cfg(self.ds_cfg, self.C)
        print(f"[PrimitiveAlignHAR] dataset_key={self.dataset_key}")
        print(f"[PrimitiveAlignHAR] channel_num={self.C}, num_class={self.num_class}")
        print(f"[PrimitiveAlignHAR] label_names={self.label_names}")
        print(f"[PrimitiveAlignHAR] channel_names={self.channel_names}")

        # ============================================================
        # 1. Load frozen VQ-VAE
        # ============================================================
        self.vq_net = IMU_VQ_Model(args)

        self.vq_ckpt_dir = self._resolve_vqvae_ckpt_dir(args)
        vq_ckpt_path = os.path.join(self.vq_ckpt_dir, "best_wrapper.pth")

        if not os.path.exists(vq_ckpt_path):
            raise FileNotFoundError(vq_ckpt_path)

        print(f"[PrimitiveAlignHAR] Loading VQ-VAE from: {vq_ckpt_path}")

        sd = torch.load(vq_ckpt_path, map_location="cpu")

        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        elif isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        elif isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]

        self.vq_net.load_state_dict(sd, strict=False)
        self.vq_net.to(self.device)
        self.vq_net.eval()

        for p in self.vq_net.parameters():
            p.requires_grad = False

        self.vq_stride = int(self.vq_net.stride_t ** self.vq_net.down_t)

        self.seq_len_pad = (
            (self.seq_len_orig + self.vq_stride - 1) // self.vq_stride
        ) * self.vq_stride

        self.P = self.seq_len_pad // self.vq_stride

        self.num_primitives = int(getattr(self.vq_net, "code_num", 512))
        self.mask_token_id = self.num_primitives

        print(
            f"[PrimitiveAlignHAR] Orig={self.seq_len_orig}, "
            f"Stride={self.vq_stride}, Padded={self.seq_len_pad}, P={self.P}, "
            f"K={self.num_primitives}"
        )

        codebook = extract_codebook(self.vq_net).float()
        self.register_buffer("codebook", codebook.to(self.device))

        self.vq_dim = int(self.codebook.shape[1])

        # ============================================================
        # 2. Load primitive profile
        # ============================================================
        profile_path = self._resolve_profile_path(args)

        print(f"[PrimitiveAlignHAR] Using primitive profile: {profile_path}")

        self.profile_path = profile_path
        self.profile = load_profile(profile_path)

        # ============================================================
        # 3. Build frozen primitive semantic embedding
        # ============================================================
        self.semantic_dim = int(getattr(args, "primitive_semantic_dim", 512))

        semantic_cache_path = os.path.join(
            os.path.dirname(profile_path),
            f"primitive_semantic_cache_{self.semantic_dim}.pt",
        )

        if os.path.exists(semantic_cache_path) and not bool(int(getattr(args, "force_build_semantic_cache", 0))):
            semantic_emb = torch.load(semantic_cache_path, map_location="cpu")
            print(f"[PrimitiveAlignHAR] Loaded semantic cache: {semantic_cache_path}")
        else:
            semantic_emb = self._build_primitive_semantic_embedding(args)
            torch.save(semantic_emb.cpu(), semantic_cache_path)
            print(f"[PrimitiveAlignHAR] Saved semantic cache: {semantic_cache_path}")

        if semantic_emb.shape[0] != self.num_primitives:
            raise ValueError(
                f"semantic_emb K={semantic_emb.shape[0]} != num_primitives={self.num_primitives}"
            )

        self.register_buffer("primitive_semantic", semantic_emb.float().to(self.device))

        # ============================================================
        # 4. Trainable alignment encoder
        # ============================================================


        # ============================================================
        # 4. Online LLM-based primitive encoder
        # ============================================================
        d_model = int(getattr(args, "d_model", 128))
        dropout = float(getattr(args, "dropout", 0.1))

        self.d_model = d_model

        # ------------------------------------------------------------
        # 4.1 Project VQ codebook embedding and primitive semantic embedding
        # ------------------------------------------------------------
        self.vq_proj = nn.Sequential(
            nn.LayerNorm(self.vq_dim),
            nn.Linear(self.vq_dim, d_model),
        )

        self.sem_proj = nn.Sequential(
            nn.LayerNorm(self.semantic_dim),
            nn.Linear(self.semantic_dim, d_model),
        )

        self.semantic_weight = float(getattr(args, "semantic_weight", 0.5))

        # ------------------------------------------------------------
        # 4.2 Load LLM for ONLINE forward
        # ------------------------------------------------------------
        llm_path = getattr(args, "llama_name", None)
        if llm_path is None:
            raise ValueError("llama_name is required because this model uses LLM in forward().")

        print(f"[PrimitiveAlignHAR] Loading ONLINE LLM encoder from: {llm_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            use_fast=False,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            output_hidden_states=True,
        ).to(self.device)

        # LLM is frozen, but still participates in forward graph.
        # Do NOT wrap LLM forward with torch.no_grad().
        for p in self.llm.parameters():
            p.requires_grad = False

        self.llm.eval()

        self.llm_hidden = int(self.llm.config.hidden_size)
        self.llm_dtype = next(self.llm.parameters()).dtype

        print(f"[PrimitiveAlignHAR] LLM hidden size = {self.llm_hidden}")
        print(f"[PrimitiveAlignHAR] LLM dtype = {self.llm_dtype}")
        # ------------------------------------------------------------
        # 4.2.1 Build textual prompt for online LLM conditioning
        # ------------------------------------------------------------
        self.use_text_prompt = bool(int(getattr(args, "use_text_prompt", 1)))

        # 大改：是否使用“当前样本统计信息”生成动态 prompt。
        # 1：每个 batch 根据 x_imu / channel_stats / primitive ids 生成 prompt。
        # 0：退回固定 task prompt。
        self.use_sample_prompt = bool(int(getattr(args, "use_sample_prompt", 1)))

        # 动态 prompt 更长，所以不要再用 128。
        self.prompt_max_length = int(getattr(args, "prompt_max_length", 384))

        # 控制动态 prompt 的长度，避免把所有通道都写进去太慢。
        self.prompt_top_channels = int(getattr(args, "prompt_top_channels", 3))
        self.prompt_num_segment_examples = int(getattr(args, "prompt_num_segment_examples", 4))
        self.prompt_include_primitive_ids = bool(int(getattr(args, "prompt_include_primitive_ids", 0)))

        if self.use_text_prompt:
            self.prompt_text = self._build_base_task_prompt()

            # 固定 prompt 作为 fallback。动态 prompt 会在 forward 里重新 tokenizer。
            prompt_enc = self.tokenizer(
                self.prompt_text,
                padding=False,
                truncation=True,
                max_length=self.prompt_max_length,
                return_tensors="pt",
            )

            self.register_buffer(
                "prompt_input_ids",
                prompt_enc["input_ids"].long(),
                persistent=True,
            )

            self.register_buffer(
                "prompt_attention_mask",
                prompt_enc["attention_mask"].long(),
                persistent=True,
            )

            self.prompt_len = int(prompt_enc["input_ids"].shape[1])

            print("[PrimitiveAlignHAR] use_text_prompt = True")
            print(f"[PrimitiveAlignHAR] use_sample_prompt = {self.use_sample_prompt}")
            print(f"[PrimitiveAlignHAR] prompt_max_length = {self.prompt_max_length}")
            print(f"[PrimitiveAlignHAR] fallback_prompt_len = {self.prompt_len}")
            print(f"[PrimitiveAlignHAR] fallback_prompt_text = {self.prompt_text}")
        else:
            self.prompt_text = ""
            self.prompt_len = 0
            self.use_sample_prompt = False

            self.register_buffer(
                "prompt_input_ids",
                torch.empty(1, 0, dtype=torch.long),
                persistent=True,
            )

            self.register_buffer(
                "prompt_attention_mask",
                torch.empty(1, 0, dtype=torch.long),
                persistent=True,
            )

            print("[PrimitiveAlignHAR] use_text_prompt = False")

        # ------------------------------------------------------------
        # 4.3 Trainable projector from primitive space to LLM embedding space
        # ------------------------------------------------------------
        self.to_llm = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.llm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.llm_hidden, self.llm_hidden),
        )

        # For masked primitive token in LLM embedding space
        self.mask_embed_llm = nn.Parameter(
            torch.randn(1, 1, self.llm_hidden) * 0.02
        )

        # For sequence classification.
        # Since LLaMA/Qwen-style CausalLM is causal, put CLS at the end,
        # so it can attend to previous primitive tokens.
        self.cls_embed_llm = nn.Parameter(
            torch.randn(1, 1, self.llm_hidden) * 0.02
        )

        # Primitive position embedding in LLM hidden space.
        # P primitive tokens + 1 final CLS token.
        self.primitive_pos_embed_llm = nn.Parameter(
            torch.randn(1, self.P + 1, self.llm_hidden) * 0.02
        )

        # ------------------------------------------------------------
        # 4.4 Heads on top of LLM hidden states
        # ------------------------------------------------------------
        self.activity_head = nn.Sequential(
            nn.LayerNorm(self.llm_hidden),
            nn.Linear(self.llm_hidden, self.num_class),
        )

        self.primitive_head = nn.Sequential(
            nn.LayerNorm(self.llm_hidden),
            nn.Linear(self.llm_hidden, self.num_primitives),
        )

        # ============================================================
        # 4.5 Channel-grounded primitive augmentation
        # ============================================================
        # Each primitive token is augmented by instance-specific local statistics.
        # stat_dim = per-channel mean/std/energy + acc_mag mean/std + gyro_mag mean/std
        self.use_segment_stats = bool(int(getattr(args, "use_segment_stats", 1)))
        self.stat_weight = float(getattr(args, "stat_weight", 1.0))
        self.stat_dim = 3 * self.C + 4

        self.stat_proj = nn.Sequential(
            nn.LayerNorm(self.stat_dim),
            nn.Linear(self.stat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Channel summary tokens preserve posture-sensitive sensor evidence.
        # Each channel gets one summary token before primitive tokens.
        self.use_channel_summary = bool(int(getattr(args, "use_channel_summary", 1)))
        # self.channel_stat_dim = 6  # mean/std/min/max/energy/abs_mean
        # mean/std/min/max/energy/abs_mean/slope
        self.channel_stat_dim = 7

        self.channel_summary_proj = nn.Sequential(
            nn.LayerNorm(self.channel_stat_dim),
            nn.Linear(self.channel_stat_dim, self.llm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.llm_hidden, self.llm_hidden),
        )

        self.channel_pos_embed_llm = nn.Parameter(
            torch.randn(1, self.C, self.llm_hidden) * 0.02
        )

        # ============================================================
        # 4.6 Semantic primitive reconstruction head
        # ============================================================
        # In addition to recovering primitive id, force masked token hidden states
        # to reconstruct the primitive semantic embedding.
        self.semantic_recon_head = nn.Sequential(
            nn.LayerNorm(self.llm_hidden),
            nn.Linear(self.llm_hidden, self.semantic_dim),
        )

        # ============================================================
        # 4.7 Label-text contrastive alignment
        # ============================================================
        self.label_align_proj = nn.Sequential(
            nn.LayerNorm(self.llm_hidden),
            nn.Linear(self.llm_hidden, self.llm_hidden),
        )

        self.label_align_temperature = float(
            getattr(args, "label_align_temperature", 0.07)
        )

        label_text_embeds = self._build_label_text_embeddings(args)
        self.register_buffer(
            "label_text_embeds",
            label_text_embeds.float().to(self.device),
            persistent=True,
        )

        # ============================================================
        # Loss weights
        # ============================================================
        self.lambda_primitive = float(getattr(args, "lambda_primitive", 0.5))
        self.lambda_semantic = float(getattr(args, "lambda_semantic", 0.2))
        self.lambda_label_align = float(getattr(args, "lambda_label_align", 0.2))
        self.align_mask_rate = float(getattr(args, "align_mask_rate", 0.3))

        print(
            f"[PrimitiveAlignHAR] use_segment_stats={self.use_segment_stats}, "
            f"stat_dim={self.stat_dim}, stat_weight={self.stat_weight}"
        )
        print(
            f"[PrimitiveAlignHAR] use_channel_summary={self.use_channel_summary}, "
            f"channel_stat_dim={self.channel_stat_dim}"
        )
        print(
            f"[PrimitiveAlignHAR] lambda_primitive={self.lambda_primitive}, "
            f"lambda_semantic={self.lambda_semantic}, "
            f"lambda_label_align={self.lambda_label_align}"
        )


    # ============================================================
    # Config helpers
    # ============================================================

    def _load_ds_cfg_from_yaml(self, args):
        ts_yaml = getattr(args, "ts_backbone_yaml", None)
        if ts_yaml is None:
            raise ValueError("ts_backbone_yaml is required.")

        if yaml is None:
            raise ImportError("pyyaml is required.")

        if os.path.exists(ts_yaml):
            config_path = ts_yaml
        else:
            project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(project_path, "configs", ts_yaml)

        with open(config_path, "r", encoding="utf-8") as f:
            cfg_all = yaml.safe_load(f)

        if self.dataset_key not in cfg_all:
            raise KeyError(f"{self.dataset_key} not found in {config_path}")

        return cfg_all[self.dataset_key]

    def _resolve_vqvae_ckpt_dir(self, args):
        vqvae_key = getattr(args, "vqvae_path", None)

        if vqvae_key is not None:
            vqvae_key = str(vqvae_key)

            if vqvae_key in self.ds_cfg:
                return self.ds_cfg[vqvae_key]

            if os.path.exists(vqvae_key):
                return vqvae_key

        ckpt_dir = getattr(args, "vqvae_ckpt_dir", None)

        if ckpt_dir is not None and os.path.exists(str(ckpt_dir)):
            return str(ckpt_dir)

        raise ValueError("Cannot resolve VQ-VAE checkpoint directory.")

    def _resolve_profile_path(self, args):
        profile_path = getattr(args, "primitive_profile_path", None)

        if profile_path is not None and os.path.exists(str(profile_path)):
            return str(profile_path)

        no_label = getattr(args, "primitive_profile_no_label_path", None)
        if no_label is not None and os.path.exists(str(no_label)):
            return str(no_label)

        candidates = [
            os.path.join(self.vq_ckpt_dir, "primitive_profiles", "primitive_profile_no_label.json"),
            os.path.join(self.vq_ckpt_dir, "primitive_profiles", "primitive_profile_with_label.json"),
            os.path.join(self.vq_ckpt_dir, "primitive_profile_strong.json"),
            os.path.join(self.vq_ckpt_dir, "primitive_profile.json"),
        ]

        for p in candidates:
            if os.path.exists(p):
                return p

        raise FileNotFoundError("No primitive profile found.")

    # ============================================================
    # Semantic embedding
    # ============================================================

    @torch.no_grad()
    def _build_primitive_semantic_embedding(self, args):
        """
        Encode primitive profile descriptions into frozen semantic embeddings.

        This is done once before training.
        """
        llm_path = getattr(args, "llama_name", None)

        if llm_path is None:
            raise ValueError("llama_name is required to build semantic embedding.")

        print(f"[PrimitiveAlignHAR] Building semantic embeddings using: {llm_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            use_fast=False,
            trust_remote_code=True,
        )

        llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            output_hidden_states=True,
        ).to(self.device).eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        texts = []

        for cid in range(self.num_primitives):
            item = self.profile.get(str(cid), None)

            if item is None:
                text = f"Primitive {cid}: an unused or unknown motion primitive."
            elif isinstance(item, str):
                text = f"Primitive {cid}: {item}"
            else:
                desc = item.get("description", f"motion primitive {cid}")
                stats = item.get("stats", {})
                dom = stats.get("dominant_channel_name", "unknown channel")
                text = (
                    f"Primitive {cid}: {desc}. "
                    f"Dominant channel: {dom}."
                )

            texts.append(text)

        batch_size = int(getattr(args, "semantic_batch_size", 16))
        all_embs = []

        for i in range(0, len(texts), batch_size):
            batch_text = texts[i:i + batch_size]

            enc = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=int(getattr(args, "semantic_max_length", 128)),
                return_tensors="pt",
            ).to(self.device)

            out = llm(
                **enc,
                output_hidden_states=True,
                use_cache=False,
            )

            h = out.hidden_states[-1].float()

            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

            all_embs.append(pooled.detach().cpu())

        semantic = torch.cat(all_embs, dim=0)

        # reduce to semantic_dim if needed
        if semantic.shape[1] != self.semantic_dim:
            # fixed random projection for cache compatibility
            torch.manual_seed(42)
            proj = torch.randn(semantic.shape[1], self.semantic_dim) / (semantic.shape[1] ** 0.5)
            semantic = semantic @ proj

        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return semantic

    # ============================================================
    # Primitive extraction
    # ============================================================

    @torch.no_grad()
    def _tokenize_imu(self, x_imu, padding_mask=None):
        x_imu = x_imu.to(self.device).float()

        B, L, C = x_imu.shape

        if C != self.C:
            raise ValueError(f"Input C={C}, expected C={self.C}")

        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device).bool()

        x_pad, _ = pad_to_multiple(x_imu, self.vq_stride, pad_value=0.0)

        if x_pad.shape[1] != self.seq_len_pad:
            if x_pad.shape[1] > self.seq_len_pad:
                x_pad = x_pad[:, :self.seq_len_pad, :]
                if padding_mask is not None:
                    padding_mask = padding_mask[:, :self.seq_len_pad]
            else:
                pad_len = self.seq_len_pad - x_pad.shape[1]
                x_pad = torch.cat([x_pad, x_pad.new_zeros(B, pad_len, C)], dim=1)
                if padding_mask is not None:
                    pad_m = torch.zeros(B, pad_len, device=self.device, dtype=torch.bool)
                    padding_mask = torch.cat([padding_mask, pad_m], dim=1)

        ids, valid_mask = self.vq_net.get_token_ids_with_mask(
            features=x_pad,
            padding_mask=padding_mask,
        )

        ids = ids[:, :self.P].long().to(self.device)
        valid_mask = valid_mask[:, :self.P].bool().to(self.device)

        return ids, valid_mask

    def _sample_mask(self, valid_mask):
        B, P = valid_mask.shape
        rand = torch.rand(B, P, device=self.device)
        mask = (rand < self.align_mask_rate) & valid_mask

        for b in range(B):
            if valid_mask[b].sum() > 0 and mask[b].sum() == 0:
                idx = torch.where(valid_mask[b])[0]
                chosen = idx[torch.randint(0, idx.numel(), (1,), device=self.device)]
                mask[b, chosen] = True

        return mask

    def _build_base_task_prompt(self):
        """
        Base task instruction.

        注意：
        这只是基础任务说明，不再是唯一 prompt。
        当前样本的 mean/std/energy/slope/primitive summary 会在 forward 里动态拼进去。
        """
        label_text = ", ".join(self.label_names)
        channel_text = ", ".join(self.channel_names)

        prompt = (
            "You are an expert in wearable-sensor human activity recognition. "
            "The input contains two complementary parts: "
            "(1) a sequence of discrete motion primitive embeddings extracted by a frozen VQ-VAE, "
            "and (2) sample-specific physical statistics computed from the original sensor signals. "
            f"The sensor channels are: {channel_text}. "
            "For dynamic activities, focus on temporal intensity, periodicity, and transition patterns. "
            "For static postures, focus on gravity orientation, axis-wise acceleration distribution, "
            "low-motion stability, and dominant sensor channels. "
            f"The candidate activity classes are: {label_text}. "
        )

        return prompt

    # Backward-compatible alias
    def _build_task_prompt(self):
        return self._build_base_task_prompt()

    def _safe_float(self, x, digits=4):
        try:
            return round(float(x), digits)
        except Exception:
            return 0.0

    @torch.no_grad()
    def _build_sample_prompt_texts(self, x_imu, ids, valid_mask, channel_stats, segment_stats):
        """
        Build one textual prompt per sample.

        这一步才是真正把当前样本的均值、方差、能量、趋势斜率、
        low-motion、dominant channel、primitive 分布写进 prompt。
        """
        if (not self.use_text_prompt) or (not self.use_sample_prompt):
            return None

        x_imu = x_imu.detach()
        ids = ids.detach()
        valid_mask = valid_mask.detach()
        channel_stats = channel_stats.detach()
        segment_stats = segment_stats.detach()

        B = x_imu.shape[0]
        prompt_texts = []

        base_prompt = self._build_base_task_prompt()

        for b in range(B):
            # --------------------------------------------------------
            # 1. Channel-level summary
            # channel_stats: [B, C, 7]
            # 0 mean, 1 std, 2 min, 3 max, 4 energy, 5 abs_mean, 6 slope
            # --------------------------------------------------------
            ch = channel_stats[b].float().cpu()
            energy = ch[:, 4]
            abs_mean = ch[:, 5]
            slope = ch[:, 6]

            top_k = min(self.prompt_top_channels, self.C)
            top_idx = torch.topk(energy, k=top_k).indices.tolist()

            channel_sents = []

            total_energy = self._safe_float(energy.mean().item(), 4)
            total_abs = self._safe_float(abs_mean.mean().item(), 4)
            avg_slope = self._safe_float(slope.abs().mean().item(), 4)

            low_motion_score = 1.0 / (1.0 + float(total_energy))
            low_motion_score = self._safe_float(low_motion_score, 4)

            if total_energy < 0.05:
                intensity_desc = "very low motion intensity"
            elif total_energy < 0.2:
                intensity_desc = "low motion intensity"
            elif total_energy < 1.0:
                intensity_desc = "moderate motion intensity"
            else:
                intensity_desc = "high motion intensity"

            for ci in top_idx:
                name = self.channel_names[ci] if ci < len(self.channel_names) else f"channel_{ci}"

                m = self._safe_float(ch[ci, 0].item(), 4)
                sd = self._safe_float(ch[ci, 1].item(), 4)
                mn = self._safe_float(ch[ci, 2].item(), 4)
                mx = self._safe_float(ch[ci, 3].item(), 4)
                en = self._safe_float(ch[ci, 4].item(), 4)
                ab = self._safe_float(ch[ci, 5].item(), 4)
                sl = self._safe_float(ch[ci, 6].item(), 4)

                if abs(sl) < 1e-3:
                    trend_desc = "nearly stable"
                elif sl > 0:
                    trend_desc = "increasing"
                else:
                    trend_desc = "decreasing"

                channel_sents.append(
                    f"{name}: mean={m}, std={sd}, min={mn}, max={mx}, "
                    f"energy={en}, abs_mean={ab}, slope={sl} ({trend_desc})."
                )

            # --------------------------------------------------------
            # 2. Primitive sequence summary
            # --------------------------------------------------------
            valid_ids = ids[b][valid_mask[b]].long().cpu()

            if valid_ids.numel() > 0:
                unique_ids = torch.unique(valid_ids)
                primitive_count = int(valid_ids.numel())
                unique_count = int(unique_ids.numel())
                diversity = self._safe_float(unique_count / max(primitive_count, 1), 4)

                # rough transition rate
                if primitive_count > 1:
                    transition_rate = (valid_ids[1:] != valid_ids[:-1]).float().mean().item()
                else:
                    transition_rate = 0.0

                transition_rate = self._safe_float(transition_rate, 4)

                primitive_summary = (
                    f"The primitive sequence has {primitive_count} valid primitive tokens, "
                    f"{unique_count} unique primitive codes, diversity={diversity}, "
                    f"and transition_rate={transition_rate}."
                )

                if self.prompt_include_primitive_ids:
                    shown = valid_ids[: min(16, primitive_count)].tolist()
                    primitive_summary += f" The first primitive ids are {shown}."
            else:
                primitive_summary = "The primitive sequence has no valid primitive tokens."

            # --------------------------------------------------------
            # 3. Segment-level local statistics summary
            # segment_stats: [B, P, 3*C+4]
            # Last four dims: acc_mean, acc_std, gyro_mean, gyro_std
            # --------------------------------------------------------
            seg = segment_stats[b].float().cpu()
            vm = valid_mask[b].cpu()

            segment_sents = []

            if vm.sum() > 0:
                valid_seg = seg[vm]

                acc_mag_mean = valid_seg[:, -4]
                acc_mag_std = valid_seg[:, -3]
                gyro_mag_mean = valid_seg[:, -2]
                gyro_mag_std = valid_seg[:, -1]

                acc_level = self._safe_float(acc_mag_mean.mean().item(), 4)
                acc_var = self._safe_float(acc_mag_std.mean().item(), 4)
                gyro_level = self._safe_float(gyro_mag_mean.mean().item(), 4)
                gyro_var = self._safe_float(gyro_mag_std.mean().item(), 4)

                segment_sents.append(
                    f"Across primitive segments, acceleration magnitude mean={acc_level}, "
                    f"acceleration variation={acc_var}, gyroscope magnitude mean={gyro_level}, "
                    f"and gyroscope variation={gyro_var}."
                )

                # show a few local segments
                n_show = min(self.prompt_num_segment_examples, valid_seg.shape[0])
                for si in range(n_show):
                    segment_sents.append(
                        f"Segment {si}: acc_mag_mean={self._safe_float(acc_mag_mean[si].item(), 4)}, "
                        f"acc_mag_std={self._safe_float(acc_mag_std[si].item(), 4)}, "
                        f"gyro_mag_mean={self._safe_float(gyro_mag_mean[si].item(), 4)}, "
                        f"gyro_mag_std={self._safe_float(gyro_mag_std[si].item(), 4)}."
                    )

            # --------------------------------------------------------
            # 4. Final prompt
            # --------------------------------------------------------
            sample_prompt = (
                base_prompt
                + "Sample-specific sensor summary: "
                + f"The overall signal shows {intensity_desc}; "
                + f"average channel energy={total_energy}, average absolute magnitude={total_abs}, "
                + f"average absolute slope={avg_slope}, low_motion_score={low_motion_score}. "
                + "Dominant sensor channels: "
                + " ".join(channel_sents)
                + " "
                + primitive_summary
                + " "
                + " ".join(segment_sents)
                + " "
                + "Now encode the following channel-grounded motion primitive sequence for activity classification. "
                + "Motion primitive sequence: "
            )

            prompt_texts.append(sample_prompt)

        return prompt_texts

    def _encode_prompt_texts(self, prompt_texts, batch_size):
        """
        Convert dynamic prompt texts into LLM input embeddings.

        Returns:
            prompt_embeds: [B, T, H]
            prompt_mask:   [B, T]
            prompt_len:    T
        """
        if not self.use_text_prompt:
            empty_embeds = torch.empty(
                batch_size,
                0,
                self.llm_hidden,
                device=self.device,
                dtype=self.llm_dtype,
            )
            empty_mask = torch.empty(
                batch_size,
                0,
                device=self.device,
                dtype=torch.long,
            )
            return empty_embeds, empty_mask, 0

        if prompt_texts is None:
            prompt_ids = self.prompt_input_ids.to(self.device)
            prompt_mask = self.prompt_attention_mask.to(self.device)

            prompt_embeds = self.llm.get_input_embeddings()(prompt_ids).detach()
            prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
            prompt_mask = prompt_mask.expand(batch_size, -1).long()

            return prompt_embeds, prompt_mask, prompt_embeds.shape[1]

        enc = self.tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=self.prompt_max_length,
            return_tensors="pt",
        ).to(self.device)

        prompt_embeds = self.llm.get_input_embeddings()(enc["input_ids"]).detach()
        prompt_mask = enc["attention_mask"].long()

        return prompt_embeds, prompt_mask, prompt_embeds.shape[1]

    @torch.no_grad()
    def _build_label_text_embeddings(self, args):
        """
        Build frozen LLM embeddings for activity label texts.

        These embeddings are used as semantic anchors for label-text contrastive
        alignment during Stage1.
        """
        texts = []

        for name in self.label_names:
            texts.append(
                "This wearable-sensor motion sequence corresponds to "
                f"the human activity: {name}."
            )

        batch_size = int(getattr(args, "label_text_batch_size", 16))
        all_embs = []

        self.llm.eval()

        for i in range(0, len(texts), batch_size):
            batch_text = texts[i:i + batch_size]

            enc = self.tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=int(getattr(args, "label_text_max_length", 64)),
                return_tensors="pt",
            ).to(self.device)

            out = self.llm(
                **enc,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            h = out.hidden_states[-1].float()
            mask = enc["attention_mask"].unsqueeze(-1).float()

            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            all_embs.append(pooled.detach().cpu())

        label_text_embeds = torch.cat(all_embs, dim=0)

        if label_text_embeds.shape[0] != self.num_class:
            raise ValueError(
                f"label_text_embeds shape mismatch: "
                f"{label_text_embeds.shape[0]} != num_class={self.num_class}"
            )

        print(
            f"[PrimitiveAlignHAR] Built label_text_embeds: "
            f"{tuple(label_text_embeds.shape)}"
        )

        return label_text_embeds

    def _compute_segment_stats(self, x_imu, valid_mask):
        """
        Compute instance-specific local physical statistics for each primitive token.

        x_imu:
            [B, L, C]

        return:
            segment_stats [B, P, 3*C + 4]
            where 3*C = channel mean/std/energy,
            +4 = acc_mag mean/std + gyro_mag mean/std.
        """
        x_imu = x_imu.to(self.device).float()
        B, L, C = x_imu.shape

        x_pad, _ = pad_to_multiple(x_imu, self.vq_stride, pad_value=0.0)

        if x_pad.shape[1] < self.P * self.vq_stride:
            pad_len = self.P * self.vq_stride - x_pad.shape[1]
            x_pad = torch.cat(
                [x_pad, x_pad.new_zeros(B, pad_len, C)],
                dim=1,
            )

        x_pad = x_pad[:, :self.P * self.vq_stride, :]
        seg = x_pad.view(B, self.P, self.vq_stride, C)

        seg_mean = seg.mean(dim=2)
        seg_std = seg.std(dim=2)
        seg_energy = (seg ** 2).mean(dim=2)

        # Accelerometer magnitude from first three channels if available.
        if C >= 3:
            acc_mag = torch.sqrt((seg[:, :, :, 0:3] ** 2).sum(dim=-1).clamp_min(1e-8))
            acc_mag_mean = acc_mag.mean(dim=2, keepdim=True)
            acc_mag_std = acc_mag.std(dim=2, keepdim=True)
        else:
            acc_mag_mean = seg.new_zeros(B, self.P, 1)
            acc_mag_std = seg.new_zeros(B, self.P, 1)

        # Gyroscope magnitude from channels 3:6 if available.
        if C >= 6:
            gyro_mag = torch.sqrt((seg[:, :, :, 3:6] ** 2).sum(dim=-1).clamp_min(1e-8))
            gyro_mag_mean = gyro_mag.mean(dim=2, keepdim=True)
            gyro_mag_std = gyro_mag.std(dim=2, keepdim=True)
        else:
            gyro_mag_mean = seg.new_zeros(B, self.P, 1)
            gyro_mag_std = seg.new_zeros(B, self.P, 1)

        stats = torch.cat(
            [
                seg_mean,
                seg_std,
                seg_energy,
                acc_mag_mean,
                acc_mag_std,
                gyro_mag_mean,
                gyro_mag_std,
            ],
            dim=-1,
        )

        if stats.shape[-1] != self.stat_dim:
            raise RuntimeError(
                f"segment_stats dim mismatch: got {stats.shape[-1]}, "
                f"expected {self.stat_dim}"
            )

        stats = stats * valid_mask.unsqueeze(-1).float()

        return stats

    def _compute_channel_stats(self, x_imu):
        """
        Compute one summary token per channel.

        x_imu:
            [B, L, C]

        return:
            channel_stats [B, C, 7]
            7 = mean/std/min/max/energy/abs_mean/slope.
        """
        x_imu = x_imu.to(self.device).float()
        B, L, C = x_imu.shape

        xc = x_imu.transpose(1, 2).contiguous()  # [B, C, L]

        ch_mean = xc.mean(dim=-1)
        ch_std = xc.std(dim=-1)
        ch_min = xc.min(dim=-1).values
        ch_max = xc.max(dim=-1).values
        ch_energy = (xc ** 2).mean(dim=-1)
        ch_abs = xc.abs().mean(dim=-1)

        # simple linear slope proxy: last - first normalized by length
        if L > 1:
            ch_slope = (xc[:, :, -1] - xc[:, :, 0]) / float(L - 1)
        else:
            ch_slope = torch.zeros_like(ch_mean)

        stats = torch.stack(
            [
                ch_mean,
                ch_std,
                ch_min,
                ch_max,
                ch_energy,
                ch_abs,
                ch_slope,
            ],
            dim=-1,
        )

        return stats
    def _encode_primitives(
        self,
        ids,
        valid_mask,
        mask_positions=None,
        segment_stats=None,
        channel_stats=None,
        prompt_texts=None,
    ):
        """
        Encode primitive ids with online frozen LLM.

        New input structure:
            sample-adaptive text prompt
            + channel summary tokens
            + channel-grounded primitive tokens
            + final CLS token
            -> frozen LLM
        """
        safe_ids = ids.clamp(0, self.num_primitives - 1)

        # ------------------------------------------------------------
        # 1. Primitive embedding
        # ------------------------------------------------------------
        vq_emb = F.embedding(safe_ids, self.codebook)
        sem_emb = F.embedding(safe_ids, self.primitive_semantic)

        primitive_z = self.vq_proj(vq_emb) + self.semantic_weight * self.sem_proj(sem_emb)

        # Instance-specific local physical statistics.
        if self.use_segment_stats and segment_stats is not None:
            segment_stats = segment_stats.to(self.device).float()
            primitive_z = primitive_z + self.stat_weight * self.stat_proj(segment_stats)

        # ------------------------------------------------------------
        # 2. Project primitive embedding to LLM hidden size
        # ------------------------------------------------------------
        llm_z = self.to_llm(primitive_z)

        # ------------------------------------------------------------
        # 3. Replace masked primitive positions
        # ------------------------------------------------------------
        if mask_positions is not None:
            llm_z = torch.where(
                mask_positions.unsqueeze(-1),
                self.mask_embed_llm.expand_as(llm_z),
                llm_z,
            )

        B, P, H = llm_z.shape

        # ------------------------------------------------------------
        # 4. Primitive position embedding
        # ------------------------------------------------------------
        cls = self.cls_embed_llm.expand(B, 1, -1)

        primitive_inputs = torch.cat([llm_z, cls], dim=1)

        primitive_inputs = (
            primitive_inputs
            + self.primitive_pos_embed_llm[:, :P + 1, :]
        )

        # ------------------------------------------------------------
        # 5. Optional channel summary tokens
        # ------------------------------------------------------------
        if self.use_channel_summary and channel_stats is not None:
            channel_stats = channel_stats.to(self.device).float()

            channel_inputs = self.channel_summary_proj(channel_stats)
            channel_inputs = channel_inputs + self.channel_pos_embed_llm[:, :channel_inputs.shape[1], :]

            channel_attention = torch.ones(
                B,
                channel_inputs.shape[1],
                device=self.device,
                dtype=torch.long,
            )
        else:
            channel_inputs = None
            channel_attention = None

        # ------------------------------------------------------------
        # 6. Dynamic prompt embeddings
        # ------------------------------------------------------------
        prompt_embeds, prompt_mask, prompt_len = self._encode_prompt_texts(
            prompt_texts=prompt_texts,
            batch_size=B,
        )

        parts = []
        masks = []

        if prompt_len > 0:
            parts.append(prompt_embeds)
            masks.append(prompt_mask.long())

        # ------------------------------------------------------------
        # 7. Build mixed LLM input
        # ------------------------------------------------------------
        if channel_inputs is not None:
            parts.append(channel_inputs)
            masks.append(channel_attention)
            channel_len = channel_inputs.shape[1]
        else:
            channel_len = 0

        parts.append(primitive_inputs)

        cls_valid = torch.ones(B, 1, device=self.device, dtype=torch.long)

        primitive_attention = torch.cat(
            [valid_mask.long(), cls_valid],
            dim=1,
        )

        masks.append(primitive_attention)

        inputs_embeds = torch.cat(parts, dim=1)
        attention_mask = torch.cat(masks, dim=1)

        # ------------------------------------------------------------
        # 8. Frozen LLM online forward
        # ------------------------------------------------------------
        self.llm.eval()

        inputs_embeds = inputs_embeds.to(dtype=self.llm_dtype)

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        h = out.hidden_states[-1].float()

        # ------------------------------------------------------------
        # 9. Extract primitive hidden states and CLS hidden state
        # ------------------------------------------------------------
        token_start = prompt_len + channel_len
        token_end = token_start + P
        cls_index = token_start + P

        token_h = h[:, token_start:token_end, :]
        cls_h = h[:, cls_index, :]

        return cls_h, token_h

    # ============================================================
    # Forward / classify
    # ============================================================

    def forward(self, x_imu, padding_mask=None, mode=None, labels=None, **kwargs):
        """
        Unified forward.

        Stage1:
            loss = activity CE
                 + lambda_primitive * masked primitive id recovery
                 + lambda_semantic * masked primitive semantic reconstruction
                 + lambda_label_align * label-text contrastive alignment

        Stage2:
            clean activity classification without primitive masking.
        """
        x_imu = x_imu.to(self.device).float()

        ids, valid_mask = self._tokenize_imu(x_imu, padding_mask)

        if labels is not None:
            labels = labels.to(self.device).long().view(-1)

        is_alignment_train = (
            self.training
            and labels is not None
            and mode not in ["classify", "test", "eval", "eval_no_mask", "inference"]
        )

        if is_alignment_train:
            primitive_mask = self._sample_mask(valid_mask)
        else:
            primitive_mask = torch.zeros_like(valid_mask)

        # ------------------------------------------------------------
        # New: sample-specific physical grounding
        # ------------------------------------------------------------
        segment_stats = self._compute_segment_stats(
            x_imu=x_imu,
            valid_mask=valid_mask,
        )

        channel_stats = self._compute_channel_stats(
            x_imu=x_imu,
        )

        # ------------------------------------------------------------
        # 大改：根据当前样本统计生成 sample-adaptive prompt
        # ------------------------------------------------------------
        prompt_texts = self._build_sample_prompt_texts(
            x_imu=x_imu,
            ids=ids,
            valid_mask=valid_mask,
            channel_stats=channel_stats,
            segment_stats=segment_stats,
        )

        cls_h, token_h = self._encode_primitives(
            ids=ids,
            valid_mask=valid_mask,
            mask_positions=primitive_mask,
            segment_stats=segment_stats,
            channel_stats=channel_stats,
            prompt_texts=prompt_texts,
        )

        activity_logits = self.activity_head(cls_h)

        # ============================================================
        # Stage2 / validation / test: clean classification branch
        # ============================================================
        if labels is None or mode in ["classify", "test", "eval", "eval_no_mask", "inference"]:
            return activity_logits

        # ============================================================
        # Stage1 alignment branch
        # ============================================================
        primitive_logits = self.primitive_head(token_h)

        loss_activity = F.cross_entropy(activity_logits, labels)

        # ------------------------------------------------------------
        # 1. Masked primitive id recovery
        # ------------------------------------------------------------
        if primitive_mask.sum() > 0:
            loss_primitive = F.cross_entropy(
                primitive_logits[primitive_mask],
                ids[primitive_mask],
            )
        else:
            loss_primitive = activity_logits.sum() * 0.0

        # ------------------------------------------------------------
        # 2. Masked primitive semantic reconstruction
        # ------------------------------------------------------------
        if primitive_mask.sum() > 0:
            pred_sem = self.semantic_recon_head(token_h[primitive_mask])

            target_sem = F.embedding(
                ids[primitive_mask].clamp(0, self.num_primitives - 1),
                self.primitive_semantic,
            ).detach()

            pred_sem = F.normalize(pred_sem.float(), dim=-1)
            target_sem = F.normalize(target_sem.float(), dim=-1)

            loss_semantic = 1.0 - (pred_sem * target_sem).sum(dim=-1).mean()
        else:
            loss_semantic = activity_logits.sum() * 0.0

        # ------------------------------------------------------------
        # 3. Label-text contrastive alignment
        # ------------------------------------------------------------
        cls_align = self.label_align_proj(cls_h)
        cls_align = F.normalize(cls_align.float(), dim=-1)

        label_text = F.normalize(self.label_text_embeds.float(), dim=-1)

        logits_label_align = (
            cls_align @ label_text.t()
        ) / max(self.label_align_temperature, 1e-6)

        loss_label_align = F.cross_entropy(logits_label_align, labels)

        # ------------------------------------------------------------
        # 4. Total loss
        # ------------------------------------------------------------
        loss_total = (
            loss_activity
            + self.lambda_primitive * loss_primitive
            + self.lambda_semantic * loss_semantic
            + self.lambda_label_align * loss_label_align
        )

        if self.training and not loss_total.requires_grad:
            trainable = [
                name for name, p in self.named_parameters()
                if p.requires_grad
            ]
            raise RuntimeError(
                "loss_total does not require grad. "
                f"loss_total={loss_total}, grad_fn={loss_total.grad_fn}, "
                f"num_trainable={len(trainable)}, "
                f"first_trainable={trainable[:30]}"
            )

        with torch.no_grad():
            pred = activity_logits.argmax(dim=-1)
            activity_acc = (pred == labels).float().mean()

            if primitive_mask.sum() > 0:
                primitive_pred = primitive_logits[primitive_mask].argmax(dim=-1)
                primitive_acc = (
                    primitive_pred == ids[primitive_mask]
                ).float().mean()
            else:
                primitive_acc = activity_logits.new_tensor(0.0)

            label_align_pred = logits_label_align.argmax(dim=-1)
            label_align_acc = (label_align_pred == labels).float().mean()

        metrics = {
            "loss_total": float(loss_total.detach().item()),
            "loss_activity": float(loss_activity.detach().item()),
            "loss_primitive": float(loss_primitive.detach().item()),
            "loss_semantic": float(loss_semantic.detach().item()),
            "loss_label_align": float(loss_label_align.detach().item()),
            "activity_acc": float(activity_acc.detach().item()),
            "primitive_acc": float(primitive_acc.detach().item()),
            "label_align_acc": float(label_align_acc.detach().item()),
            "mask_ratio_actual": float(
                primitive_mask.sum().detach().item()
                / max(valid_mask.sum().detach().item(), 1)
            ),
        }

        return loss_total, activity_logits, metrics


    @torch.no_grad()
    def classify(self, x_imu, padding_mask=None):
        """
        Inference only. Training code should call forward(..., mode="classify")
        instead of this function.
        """
        was_training = self.training
        self.eval()

        logits = self.forward(
            x_imu=x_imu,
            padding_mask=padding_mask,
            mode="classify",
            labels=None,
        )

        probs = F.softmax(logits, dim=-1)

        if was_training:
            self.train()

        return logits, probs
    def save_wrapper(self, path):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_wrapper(self, path, map_location="cpu"):
        sd = torch.load(path, map_location=map_location)
        self.load_state_dict(sd, strict=True)