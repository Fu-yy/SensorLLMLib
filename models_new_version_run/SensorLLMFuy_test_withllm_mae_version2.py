# code1_stageC_minimal.py
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Primitive builder (P1): energy + jerk -> K bins
# -------------------------
@torch.no_grad()
def primitive_ids_energy_jerk(x_norm: torch.Tensor, patch_len: int, K: int = 32) -> torch.Tensor:
    """
    x_norm: [B,L,C]
    return: pid [B,P] in [0..K-1]
    """
    B, L, C = x_norm.shape
    assert L % patch_len == 0
    P = L // patch_len
    xp = x_norm.view(B, P, patch_len, C)

    energy = (xp ** 2).mean(dim=(2, 3))  # [B,P]
    jerk = torch.zeros_like(energy)
    jerk[:, 1:] = (energy[:, 1:] - energy[:, :-1]).abs()

    feat = torch.stack([energy, jerk], dim=-1)  # [B,P,2]
    feat = (feat - feat.mean(dim=(0, 1), keepdim=True)) / (feat.std(dim=(0, 1), keepdim=True) + 1e-6)
    score = feat[..., 0] + 0.5 * feat[..., 1]  # [B,P]

    # balanced bins by quantiles (NOTE: per-batch quantile; for paper use offline fixed quantiles)
    q = torch.quantile(score.flatten(), torch.linspace(0, 1, K + 1, device=score.device))
    pid = torch.bucketize(score, q[1:-1])  # [B,P]
    pid = pid.clamp(0, K - 1)
    return pid


def _pick_contiguous_span(P: int, span_len: int, device) -> Tuple[int, int]:
    span_len = max(1, min(P, int(span_len)))
    s = int(torch.randint(0, max(1, P - span_len + 1), (1,), device=device).item())
    e = s + span_len
    return s, e


def _majority_vote(ids_1d: torch.Tensor, K: int) -> int:
    if ids_1d.numel() == 0:
        return 0
    cnt = torch.bincount(ids_1d.long().clamp(0, K - 1), minlength=K)
    return int(cnt.argmax().item())


@dataclass
class TeacherOut:
    probs: torch.Tensor  # [B,K]
    ok: torch.Tensor     # [B] bool


class PrimitiveLLMTeacher(nn.Module):
    """
    Frozen causal LLM teacher.
    Prompt ends with "Answer:" and teacher returns p(AnswerToken in {P0..P(K-1)}).
    IMPORTANT: P0..P(K-1) are added as special tokens. Their embeddings are randomly init unless adapted.
    """
    def __init__(self, llm_name_or_path: str, K: int, device: torch.device):
        super().__init__()
        self.K = K
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        prim_tokens = [f"P{i}" for i in range(K)] + ["[PMASK]"]
        add = self.tokenizer.add_special_tokens({"additional_special_tokens": prim_tokens})

        self.llm = AutoModelForCausalLM.from_pretrained(llm_name_or_path)
        if add > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))

        self.llm.to(device)
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad = False

        self.prim_token_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(f"P{i}") for i in range(K)],
            dtype=torch.long,
            device=device,
        )

    @torch.no_grad()
    def forward(self, prompts: List[str], max_length: int = 512) -> TeacherOut:
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        out = self.llm(**enc)  # logits [B,T,V]
        logits = out.logits

        # ---- FIX: take per-sample last non-pad token position ----
        # attention_mask [B,T], last_idx [B]
        attn = enc["attention_mask"]
        last_idx = attn.sum(dim=1) - 1  # [B]
        last_idx = last_idx.clamp_min(0)

        # gather logits at last_idx: [B,V]
        B = logits.size(0)
        last = logits[torch.arange(B, device=logits.device), last_idx, :]

        sel = last.index_select(dim=-1, index=self.prim_token_ids)  # [B,K]
        probs = torch.softmax(sel, dim=-1)
        ok = torch.ones(probs.size(0), device=probs.device, dtype=torch.bool)
        return TeacherOut(probs=probs, ok=ok)


def build_teacher_prompt(pid_seq_1d: torch.Tensor,
                         s: int, e: int,
                         sample_rate: int,
                         missing_channels: str,
                         K: int) -> str:
    """
    pid_seq_1d: [P]
    Replace [s:e] with [PMASK] once (single span token)
    """
    tokens = []
    for i, k in enumerate(pid_seq_1d.tolist()):
        if s <= i < e:
            if (len(tokens) == 0) or (tokens[-1] != "[PMASK]"):
                tokens.append("[PMASK]")
        else:
            tokens.append(f"P{int(k)}")

    seq_txt = " ".join(tokens)

    prompt = (
        "You are an expert in wearable sensor activity modeling.\n"
        f"Observed primitive sequence: {seq_txt}\n"
        f"Device info: sample_rate={int(sample_rate)}Hz, missing_channels={missing_channels}\n"
        "Task: Infer the missing primitive category represented by [PMASK].\n"
        f"Answer with exactly one token among P0..P{K-1}.\n"
        "Answer:"
    )
    return prompt


# -------------------------
# Minimal PatchEmbed/Resampler/Decoder for runnable demo
# (Replace with your own modules later; keep interfaces.)
# -------------------------
class DummyPatchEmbed(nn.Module):
    def __init__(self, P: int, D: int):
        super().__init__()
        self.P = P
        self.D = D
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D))

    def forward(self, x_norm: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x_norm: [B,L,C] -> naive patchify average into [B,P,D]
        B, L, C = x_norm.shape
        z = x_norm.view(B, self.P, L // self.P, C).mean(dim=(2, 3), keepdim=False)  # [B,P]
        z = z.unsqueeze(-1).repeat(1, 1, self.D)  # [B,P,D]
        if patch_mask is not None:
            w = patch_mask.unsqueeze(-1).float()
            z = z * (1 - w) + self.mask_token * w
        return z


class DummyResampler(nn.Module):
    def __init__(self, D: int, M: int):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, M, D))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        # simplest: return learnable latents (ignore z) just for runnable shape
        return self.latents.expand(B, -1, -1)


class DummyDecoder(nn.Module):
    def __init__(self, P: int, out_dim: int, D: int):
        super().__init__()
        self.P = P
        self.proj = nn.Linear(D, out_dim)

    def forward(self, lat: torch.Tensor) -> torch.Tensor:
        # lat [B,M,D] -> mean -> [B, out_dim] -> repeat P
        x = lat.mean(dim=1)
        pred = self.proj(x).unsqueeze(1).repeat(1, self.P, 1)
        return pred


def patch_rfft_amp(x_patch: torch.Tensor, patch_len: int) -> torch.Tensor:
    # x_patch: [B,P,patch_len*C] -> amp [B,P,C,F]
    B, P, D = x_patch.shape
    C = D // patch_len
    xt = x_patch.view(B * P, patch_len, C).transpose(1, 2)
    amp = torch.fft.rfft(xt.float(), dim=-1).abs()
    Fbins = amp.size(-1)
    return amp.view(B, P, C, Fbins)


class StageCModel_Code1(nn.Module):
    def __init__(self, seq_len=200, patch_len=20, C=15, D=64, M=16, K=32,
                 mask_rate=0.5, prim_span_len=2,
                 lambda_prim_ce=0.2, lambda_prim_kl=0.5,
                 lambda_freq=0.3,
                 use_llm_teacher=False, teacher_path=""):
        super().__init__()
        assert seq_len % patch_len == 0
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.P = seq_len // patch_len
        self.C = C
        self.D = D
        self.M = M
        self.K = K
        self.mask_rate = mask_rate
        self.prim_span_len = prim_span_len
        self.lambda_prim_ce = lambda_prim_ce
        self.lambda_prim_kl = lambda_prim_kl
        self.lambda_freq = lambda_freq

        # demo modules (replace with yours)
        self.patch_embed = DummyPatchEmbed(P=self.P, D=self.D)
        self.resampler = DummyResampler(D=self.D, M=self.M)
        self.decoder = DummyDecoder(P=self.P, out_dim=self.patch_len * self.C, D=self.D)

        self.prim_head = nn.Sequential(nn.LayerNorm(self.D), nn.Linear(self.D, self.K))

        self.sample_rate = 50
        self.missing_channels = "none"

        self.teacher = None
        if use_llm_teacher and teacher_path:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.teacher = PrimitiveLLMTeacher(teacher_path, K=self.K, device=dev)

    def block_patch_mask(self, B, P, mask_rate, device):
        k = max(1, int(P * mask_rate))
        mask = torch.zeros(B, P, device=device, dtype=torch.bool)
        for b in range(B):
            s = int(torch.randint(0, max(1, P - k + 1), (1,), device=device).item())
            e = min(P, s + k)
            mask[b, s:e] = True
            if mask[b].all():
                mask[b, torch.randint(0, P, (1,), device=device)] = False
        return mask

    def forward(self, x: torch.Tensor):
        """
        x: [B,L,C]
        returns total_loss, union_mask, info
        """
        B, L, C = x.shape
        assert L == self.seq_len and C == self.C

        # instance norm
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True).clamp_min(1e-5)
        x_norm = (x - mu) / sigma

        # (A) primitive ids on clean
        pid = primitive_ids_energy_jerk(x_norm, patch_len=self.patch_len, K=self.K)  # [B,P]

        # (B) missing span (prim_mask)
        prim_mask = torch.zeros(B, self.P, device=x.device, dtype=torch.bool)
        miss_label = torch.zeros(B, device=x.device, dtype=torch.long)
        span_s = torch.zeros(B, device=x.device, dtype=torch.long)
        span_e = torch.zeros(B, device=x.device, dtype=torch.long)
        for b in range(B):
            s, e = _pick_contiguous_span(self.P, self.prim_span_len, x.device)
            prim_mask[b, s:e] = True
            span_s[b] = s
            span_e[b] = e
            miss_label[b] = _majority_vote(pid[b, s:e], K=self.K)

        # (C) MAE mask
        mae_mask = self.block_patch_mask(B, self.P, self.mask_rate, x.device)
        union_mask = mae_mask | prim_mask

        # (D) encode
        z = self.patch_embed(x_norm, patch_mask=union_mask)     # [B,P,D]
        lat = self.resampler(z)                                # [B,M,D]

        # (E) recon
        pred = self.decoder(lat)                               # [B,P,patch_len*C]
        target = x_norm.view(B, self.P, self.patch_len * self.C).float()
        pred = pred.float()

        loss_time = (pred - target).pow(2).mean(dim=-1)         # [B,P]
        amp_pred = patch_rfft_amp(pred, self.patch_len)
        amp_tgt = patch_rfft_amp(target, self.patch_len)
        loss_freq = (amp_pred - amp_tgt).abs().mean(dim=(-1, -2))  # [B,P]
        loss_patch = loss_time + self.lambda_freq * loss_freq
        loss_mae = (loss_patch * mae_mask.float()).sum() / (mae_mask.float().sum() + 1e-6)

        # (F) missing primitive student head
        pooled = lat.mean(dim=1)            # [B,D]
        stu_logits = self.prim_head(pooled) # [B,K]
        loss_prim_ce = F.cross_entropy(stu_logits, miss_label)

        # (G) teacher KL (optional)
        loss_prim_kl = torch.zeros((), device=x.device)
        if self.teacher is not None:
            prompts = []
            for b in range(B):
                prompts.append(build_teacher_prompt(
                    pid_seq_1d=pid[b].detach().cpu(),
                    s=int(span_s[b].item()),
                    e=int(span_e[b].item()),
                    sample_rate=self.sample_rate,
                    missing_channels=self.missing_channels,
                    K=self.K
                ))
            t_out = self.teacher(prompts)
            t_probs = t_out.probs.to(x.device).clamp_min(1e-8)
            stu_logp = F.log_softmax(stu_logits, dim=-1)
            loss_prim_kl = F.kl_div(stu_logp, t_probs, reduction="batchmean")

        total = loss_mae + self.lambda_prim_ce * loss_prim_ce + self.lambda_prim_kl * loss_prim_kl
        info = {
            "loss_mae": float(loss_mae.item()),
            "loss_prim_ce": float(loss_prim_ce.item()),
            "loss_prim_kl": float(loss_prim_kl.item()),
            "mask_rate_mae": float(mae_mask.float().mean().item()),
            "mask_rate_prim": float(prim_mask.float().mean().item()),
        }
        return total, union_mask, info


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, C = 2, 200, 15
    x = torch.randn(B, L, C)

    model = StageCModel_Code1(
        seq_len=L, patch_len=20, C=C, D=64, M=16, K=32,
        use_llm_teacher=False, teacher_path=""
    )
    loss, mask, info = model(x)
    print("loss:", loss.item())
    print("mask shape:", mask.shape, "info:", info)
