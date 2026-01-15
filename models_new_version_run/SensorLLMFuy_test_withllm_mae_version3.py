import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# 禁用 fused attention 以兼容性优先
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

try:
    import yaml
except Exception:
    yaml = None


# ============================================================
# 0. Helper & VQ Component (The Core of Strategy P2)
# ============================================================

class VectorQuantizer(nn.Module):
    """
    [Strategy P2 Implementation]
    Learnable Codebook to discretize continuous patch embeddings into Primitives.
    This acts as an 'Online K-Means'.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings  # K (Primitives count)
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # The Codebook (Cluster Centers)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs: torch.Tensor):
        # inputs: [B, P, D]
        # returns: quantized [B, P, D], indices [B, P], loss

        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)  # [B*P, D]

        # Calculate distances: (x-y)^2 = x^2 + y^2 - 2xy
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding: get closest codebook index (Primitive ID)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [B*P, 1]
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)  # [B, P, D]
        indices = encoding_indices.view(input_shape[0], input_shape[1])  # [B, P]

        # Loss: 1. Codebook loss (update dictionary) 2. Commitment loss (encoder output stays close to dictionary)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator (Gradient copy)
        quantized = inputs + (quantized - inputs).detach()

        return quantized, indices, loss

    def get_codebook_indices(self, inputs):
        # Helper for inference only
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices.view(inputs.shape[0], inputs.shape[1])


def patch_rfft_amp(x_patch, patch_len):
    B, P, D = x_patch.shape
    C = D // patch_len
    xt = x_patch.view(B * P, patch_len, C).transpose(1, 2)
    amp = torch.fft.rfft(xt.float(), dim=-1).abs()
    return amp


# ============================================================
# 1. Scientific Components (LLM Teacher)
# ============================================================

@torch.no_grad()
def build_teacher_prompt(pid_seq: torch.Tensor,
                         s: int, e: int,
                         sample_rate: int,
                         missing_channels: str) -> str:
    """
    Construct prompt for the LLM based on Primitive IDs.
    """
    tokens = []
    # pid_seq is CPU tensor [P]
    for i, k in enumerate(pid_seq.tolist()):
        if s <= i < e:
            # Insert [PMASK] only once for the contiguous block if we want block masking
            # Or insert per token. Here we simulate block masking in text:
            if (len(tokens) == 0) or (tokens[-1] != "[PMASK]"):
                tokens.append("[PMASK]")
            # If we want 1-to-1 mapping, remove the check above. 
            # But "Block" implies one semantic gap. Let's stick to one token for the gap.
        else:
            tokens.append(f"P{int(k)}")

    seq_txt = " ".join(tokens)
    prompt = (
        "You are an expert in wearable sensor activity modeling.\n"
        f"Observed primitives: {seq_txt}\n"
        f"Device info: sample_rate={int(sample_rate)}Hz, missing_channels={missing_channels}\n"
        "Task: Predict the missing primitive category for the masked section.\n"
        "Answer with exactly one token among P0..P31.\n"
        "Answer:"
    )
    return prompt


class PrimitiveLLMTeacher(nn.Module):
    """
    Frozen causal LLM teacher.
    """

    def __init__(self, llm_name_or_path: str, K: int, device: torch.device):
        super().__init__()
        self.K = K
        self.device = device

        print(f"Loading LLM Teacher from {llm_name_or_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add primitive tokens + PMASK
        prim_tokens = [f"P{i}" for i in range(K)] + ["[PMASK]"]
        # Important: Check if tokens exist before adding to avoid ballooning vocab on re-runs
        num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": prim_tokens})

        self.llm = AutoModelForCausalLM.from_pretrained(llm_name_or_path, torch_dtype=torch.float16)
        if num_added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))

        self.llm.to(device)
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad = False

        # Cache IDs for P0...PK-1
        self.prim_token_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(f"P{i}") for i in range(K)],
            dtype=torch.long,
            device=device,
        )

    @torch.no_grad()
    def forward(self, prompts: list[str]) -> torch.Tensor:
        """
        Return: probs [B, K] (Normalized probability over primitive tokens)
        """
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        out = self.llm(**enc)
        logits = out.logits[:, -1, :]  # [B, vocab]

        # Extract only primitive token logits
        prim_logits = logits.index_select(dim=-1, index=self.prim_token_ids)  # [B, K]
        probs = torch.softmax(prim_logits, dim=-1)  # [B, K]
        return probs


# ============================================================
# 2. Backbone Components (Resampler, Decoder, Embed)
# ============================================================

class ResamplerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln_latents = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.ln_cross_q = nn.LayerNorm(hidden_size)
        self.ln_cross_kv = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.ln_ffn = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, latents: torch.Tensor, sensor_embeds: torch.Tensor) -> torch.Tensor:
        q_sa = self.ln_latents(latents)
        latents_out, _ = self.self_attn(q_sa, q_sa, q_sa)
        latents = latents + latents_out

        q_ca = self.ln_cross_q(latents)
        k_ca = v_ca = self.ln_cross_kv(sensor_embeds)
        cross_out, _ = self.cross_attn(query=q_ca, key=k_ca, value=v_ca)
        latents = latents + cross_out

        ffn_out = self.ffn(self.ln_ffn(latents))
        latents = latents + ffn_out
        return latents


class DeepResampler(nn.Module):
    def __init__(self, hidden_size: int, num_latents: int, depth: int = 4, num_heads: int = 8):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_size))
        nn.init.trunc_normal_(self.latents, std=0.02)
        self.layers = nn.ModuleList([ResamplerLayer(hidden_size, num_heads) for _ in range(depth)])
        self.final_ln = nn.LayerNorm(hidden_size)

    def forward(self, sensor_embeds: torch.Tensor) -> torch.Tensor:
        B = sensor_embeds.shape[0]
        x = self.latents.expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, sensor_embeds)
        return self.final_ln(x)


class MAEDecoder(nn.Module):
    def __init__(self, hidden_size: int, num_patches: int, patch_dim: int, num_heads: int = 4, depth: int = 2):
        super().__init__()
        self.pos_queries = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_queries, std=0.02)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True, norm_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.pred_head = nn.Linear(hidden_size, patch_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        B = latents.shape[0]
        queries = self.pos_queries.expand(B, -1, -1)
        x = queries
        for layer in self.layers:
            x = layer(tgt=x, memory=latents)
        x = self.norm(x)
        return self.pred_head(x)


class TinyTransHead(nn.Module):
    def __init__(self, d, nhead=8, depth=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=4 * d, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        return self.ln(self.enc(x))


class AttnPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, max(1, dim // 2)), nn.GELU(),
                                   nn.Linear(max(1, dim // 2), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x), dim=1)
        return (x * w).sum(dim=1)


class PatchEmbeddingConv(nn.Module):
    def __init__(self, seq_len: int, patch_len: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.embed_dim = embed_dim
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim // 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.proj = nn.Linear(patch_len * embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        h = self.stem(x.transpose(1, 2)).transpose(1, 2)
        h = torch.reshape(h, (B, self.num_patches, self.patch_len * self.embed_dim))
        h = self.proj(h)
        h = self.norm(h)
        h = h + self.pos_embed

        # Return FULL embeddings (before masking) for VQ/Teacher
        full_embeddings = h.clone()

        if patch_mask is not None:
            mask_tokens = self.mask_token.expand(B, self.num_patches, self.embed_dim)
            w = patch_mask.unsqueeze(-1).type_as(h)
            h = h * (1 - w) + mask_tokens * w

        return h, full_embeddings


# ============================================================
# 3. Main Model (P2 Integrated)
# ============================================================

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load Config (Dataset Specifics)
        self.dataset_key = str(getattr(args, "dataset_key", getattr(args, "data", "mhealth"))).lower()
        self.ds_cfg: Dict[str, Any] = {}
        if hasattr(args, "ds_cfg") and isinstance(args.ds_cfg, dict):
            self.ds_cfg = args.ds_cfg
        else:
            ts_yaml = getattr(args, "ts_backbone_yaml", None)
            if ts_yaml is not None and os.path.exists(ts_yaml):
                with open(ts_yaml, "r", encoding="utf-8") as f:
                    cfg_all = yaml.safe_load(f)
                self.ds_cfg = cfg_all.get(self.dataset_key, {})

        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 15)))
        self.num_class = int(self.ds_cfg.get("num_labels", getattr(args, "num_class", 12)))
        self.stage = int(getattr(args, "stage", 1))

        self.seq_len = int(getattr(args, "seq_len", 200))
        self.patch_len = int(getattr(args, "patch_len", 20))
        self.P = self.seq_len // self.patch_len
        self.mask_rate = float(getattr(args, "mask_rate", 0.5))

        self.resampler_dim = int(getattr(args, "resampler_dim", 256))
        self.num_latents = int(getattr(args, "num_latents", min(self.P, 16)))

        # --- Core Components ---
        self.patch_embed = PatchEmbeddingConv(
            seq_len=self.seq_len, patch_len=self.patch_len, in_channels=self.C, embed_dim=self.resampler_dim
        )
        self.resampler = DeepResampler(
            hidden_size=self.resampler_dim, num_latents=self.num_latents,
            depth=int(getattr(args, "resampler_depth", 4)), num_heads=int(getattr(args, "num_heads", 8))
        )
        self.mae_decoder = MAEDecoder(
            hidden_size=self.resampler_dim, num_patches=self.P,
            patch_dim=self.patch_len * self.C, depth=int(getattr(args, "dec_depth", 2))
        )

        self.head = TinyTransHead(self.resampler_dim)
        self.pool = AttnPool(self.resampler_dim)
        self.cls_head = nn.Linear(self.resampler_dim, self.num_class)
        self.norm_eps = float(getattr(args, "norm_eps", 1e-5))
        self.loss_fp32 = bool(getattr(args, "loss_fp32", True))

        # ====================================================
        # STRATEGY P2: VECTOR QUANTIZATION & LLM COMPONENTS
        # ====================================================
        self.num_primitives = int(getattr(args, "num_primitives", 32))  # K

        # 1. The Vector Quantizer (The "P2" clustering engine)
        self.quantizer = VectorQuantizer(num_embeddings=self.num_primitives, embedding_dim=self.resampler_dim,
                                         commitment_cost=0.25)

        # 2. Missing Primitive Predictor (The "Student" Head)
        # Input: Pooled features from Resampler/TransHead
        # Output: Logits for the missing primitive ID
        self.prim_head = nn.Sequential(
            nn.Linear(self.resampler_dim, self.resampler_dim),
            nn.GELU(),
            nn.Linear(self.resampler_dim, self.num_primitives)
        )

        # 3. LLM Teacher (Frozen)
        self.use_llm_teacher = bool(getattr(args, "use_llm_teacher", False))
        self.teacher_llm_path = str(getattr(args, "teacher_llm_path", r'D:\fuy\MyCode\Llama-3.2-1B')).strip()
        self.teacher = None

        # Parameters for prompt construction
        self.sample_rate = int(getattr(args, "sample_rate", int(self.ds_cfg.get("sample_rate", 50))))
        self.missing_channels = str(getattr(args, "missing_channels", "none"))
        self.prim_span_len = int(getattr(args, "prim_span_len", 2))  # Length of missing span for LLM task
        self.lambda_prim_ce = float(getattr(args, "lambda_prim_ce", 0.2))
        self.lambda_prim_kl = float(getattr(args, "lambda_prim_kl", 0.5))

        # Lazy Init / Init Logic
        if self.use_llm_teacher and self.teacher_llm_path and self.stage == 1:
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.teacher = PrimitiveLLMTeacher(
                    llm_name_or_path=self.teacher_llm_path,
                    K=self.num_primitives,
                    device=device
                )
            except Exception as e:
                print(f"[Warning] Failed to load LLM Teacher: {e}. Training will proceed without distillation.")

    def _align_seq_len(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        if L != self.seq_len:
            if L > self.seq_len:
                x = x[:, -self.seq_len:, :]
            else:
                pad = self.seq_len - L
                x = torch.cat([torch.zeros(B, pad, C, device=x.device, dtype=x.dtype), x], dim=1)
        return x

    def _pick_contiguous_span_mask(self, B, P, span_len, device):
        """Generate a mask for a contiguous span of primitives (for Stage C task)"""
        mask = torch.zeros(B, P, device=device, dtype=torch.bool)
        span_starts = torch.zeros(B, device=device, dtype=torch.long)
        span_ends = torch.zeros(B, device=device, dtype=torch.long)

        for b in range(B):
            s = torch.randint(0, max(1, P - span_len + 1), (1,), device=device).item()
            e = min(P, s + span_len)
            mask[b, s:e] = True
            span_starts[b] = s
            span_ends[b] = e
        return mask, span_starts, span_ends

    def block_patch_mask(self, B, P, mask_rate, device):
        """Original MAE block masking"""
        k = max(1, int(P * mask_rate))
        mask = torch.zeros(B, P, device=device, dtype=torch.bool)
        for b in range(B):
            s = torch.randint(0, P, (1,), device=device).item()
            e = min(P, s + k)
            mask[b, s:e] = True
            if mask[b].all(): mask[b, torch.randint(0, P, (1,), device=device)] = False
        return mask

    def forward(self, batch_x, padding_mask=None, mode: Optional[str] = None):
        if mode is None:
            mode = "pretrain" if self.stage == 1 else "classify"

        if not torch.is_tensor(batch_x):
            batch_x = torch.as_tensor(batch_x)
        x = self._align_seq_len(batch_x)

        # Instance Norm
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True).clamp_min(self.norm_eps)
        x_norm = (x - mu) / sigma

        # ==========================================
        # Branch 1: Downstream Classification (Stage 2)
        # ==========================================
        if mode == "classify":
            # No masking, full inference
            z_masked, _ = self.patch_embed(x_norm, patch_mask=None)  # z_masked is full here

            if bool(getattr(self.args, "no_resampler", False)):
                lat = z_masked
            else:
                lat = self.resampler(z_masked)

            lat = self.head(lat)
            pooled = lat.mean(dim=1) if lat.dim() == 3 else lat
            logits = self.cls_head(pooled)
            return logits

        # ==========================================
        # Branch 2: Pretraining with P2 (Stage 1 / Stage C)
        # ==========================================
        B = x_norm.size(0)

        # ---------------------------------------------------------
        # Step 1: "Teacher Path" - Get Ground Truth Primitives (P2)
        # ---------------------------------------------------------
        # Use full x_norm to get embeddings, then quantize them to get IDs
        with torch.no_grad():
            _, z_full_no_grad = self.patch_embed(x_norm, patch_mask=None)

        # Note: We pass z_full to quantizer. In P2, we want the quantizer codebook to learn.
        # But usually VQ-VAE updates codebook on the *forward* pass. 
        # Here we use z_full (detached in teacher path logic, but we need gradients for VQ codebook adaptation).
        # So we re-compute z_full with grads allowed for VQ training.
        _, z_full = self.patch_embed(x_norm, patch_mask=None)

        # Run VQ: z_q is quantized embeddings, indices is primitive IDs [B, P]
        z_q, primitive_ids, loss_vq = self.quantizer(z_full)

        # ---------------------------------------------------------
        # Step 2: Generate Masks (MAE Mask + Primitive Reasoning Mask)
        # ---------------------------------------------------------
        # A. Standard MAE Mask (Random Blocks)
        mae_mask = self.block_patch_mask(B, self.P, self.mask_rate, x.device)

        # B. Stage-C Reasoning Mask (Contiguous Span for LLM Task)
        prim_mask, span_s, span_e = self._pick_contiguous_span_mask(B, self.P, self.prim_span_len, x.device)

        # Identify the "Correct Answer" for the missing span (Majority Vote or just the IDs)
        # For simplicity and robustness, we predict the ID of the *center* of the masked span 
        # or we try to predict the distribution of the whole span. 
        # Let's simplify: The student tries to predict the Primitive ID of the masked tokens.
        # We target the specific primitive IDs masked by prim_mask.
        target_pids = primitive_ids.detach().clone()  # [B, P]

        # Union Mask
        union_mask = mae_mask | prim_mask

        # ---------------------------------------------------------
        # Step 3: "Student Path" - Masked Encoding & Reconstruction
        # ---------------------------------------------------------
        # Pass masked input
        z_masked, _ = self.patch_embed(x_norm, patch_mask=union_mask)
        lat = self.resampler(z_masked)

        # A. Reconstruction (Signal Level) - via MAEDecoder
        # MAE Decoder usually takes resampler output
        pred_signal = self.mae_decoder(lat)  # [B, P, patch_len*C]

        # ---------------------------------------------------------
        # Step 4: Loss Calculation
        # ---------------------------------------------------------

        # --- Loss 1: Reconstruction (Time + Freq) ---
        target_signal = x_norm.view(B, self.P, self.patch_len * x_norm.size(2))
        if self.loss_fp32:
            pred_signal = pred_signal.float()
            target_signal = target_signal.float()

        loss_time = (pred_signal - target_signal).pow(2).mean(dim=-1)
        amp_pred = patch_rfft_amp(pred_signal, self.patch_len)
        amp_tgt = patch_rfft_amp(target_signal, self.patch_len)
        loss_freq = (amp_pred - amp_tgt).abs().mean(dim=(-1, -2)).view(B, self.P)

        lam_f = float(getattr(self.args, "lambda_freq", 0.3))
        loss_recon = (loss_time + lam_f * loss_freq)
        # Only supervise on MAE mask parts (traditional MAE logic)
        loss_recon = (loss_recon * mae_mask.float()).sum() / (mae_mask.float().sum() + 1e-6)

        # --- Loss 2: Primitive Prediction (Student Classification) ---
        # Student predicts primitive logits from latent representation
        # Use pooled latents from TinyTransHead for classification
        lat_trans = self.head(lat)
        # Or simpler: project latents to P predictions? 
        # To align with P2 (Masked Patch Prediction), we need [B, P, K].
        # But Resampler output is [B, M, D]. We need a way to map back to P or use a global prediction.
        # Strategy: Use the MAE Decoder features? Or just pool?
        # Better: Let's assume we predict the *Missing Span's Primitive* using a pooled vector.
        pooled = self.pool(lat_trans)  # [B, D]
        stu_logits = self.prim_head(pooled)  # [B, K] (Global prediction for the "event")

        # Ground Truth for Global Prediction: Majority ID in the missing span
        # This aligns with "What happened in this missing gap?"
        span_targets = []
        for b in range(B):
            ids_in_span = primitive_ids[b, span_s[b]:span_e[b]]
            if len(ids_in_span) > 0:
                mode_id = torch.mode(ids_in_span).values
            else:
                mode_id = torch.tensor(0, device=x.device)
            span_targets.append(mode_id)
        span_targets = torch.stack(span_targets)  # [B]

        loss_prim_ce = F.cross_entropy(stu_logits, span_targets)

        # --- Loss 3: LLM Distillation (Optional Stage C) ---
        loss_distill = torch.tensor(0.0, device=x.device)
        if self.teacher is not None:
            # Build prompts using the Ground Truth Primitive IDs (from VQ)
            prompts = []
            pid_list = primitive_ids.detach().cpu()
            for b in range(B):
                prompts.append(
                    build_teacher_prompt(
                        pid_seq=pid_list[b],
                        s=int(span_s[b].item()),
                        e=int(span_e[b].item()),
                        sample_rate=self.sample_rate,
                        missing_channels=self.missing_channels
                    )
                )

            # Teacher Forward
            teacher_probs = self.teacher(prompts).to(x.device)  # [B, K]

            # KL Loss
            stu_log_probs = F.log_softmax(stu_logits, dim=-1)
            loss_distill = F.kl_div(stu_log_probs, teacher_probs, reduction='batchmean')

        # Total Loss
        total_loss = loss_recon + loss_vq + self.lambda_prim_ce * loss_prim_ce + self.lambda_prim_kl * loss_distill

        return total_loss, union_mask, {
            "loss_recon": float(loss_recon.item()),
            "loss_vq": float(loss_vq.item()),
            "loss_prim": float(loss_prim_ce.item()),
            "loss_distill": float(loss_distill.item())
        }