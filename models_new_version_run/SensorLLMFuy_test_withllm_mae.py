import os
import json
import math
import random
import hashlib
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml
except Exception:
    yaml = None

# optional: online LLM teacher
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None
import torch



import os
import json
import hashlib
import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM



# llm_tokens.py
from typing import List, Tuple
import torch


import torch
import torch.nn.functional as F

@torch.no_grad()
def debug_teacher_alignment(teacher, prompt: str):
    tok = teacher.tokenizer
    dev = teacher._input_device()
    if not prompt.endswith(" "):
        prompt += " "

    enc = tok([prompt], return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(dev)
    attn_mask = enc["attention_mask"].to(dev)

    out = teacher.model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits  # [1,T,V]
    last_idx = attn_mask.sum(dim=1) - 1
    last_logits = logits[0, int(last_idx.item())]  # [V]
    probs = torch.softmax(last_logits, dim=-1)

    cand_ids = teacher.cand_ids_cpu.to(dev)
    cand_probs = probs.index_select(0, cand_ids)

    # metrics
    cand_mass = float(cand_probs.sum().item())
    p = cand_probs / (cand_probs.sum() + 1e-12)
    entropy = float(-(p * (p + 1e-12).log()).sum().item())
    top2 = torch.topk(p, k=2).values
    margin = float((top2[0] - top2[1]).item())

    # print("[TeacherAlign] cand_mass=", cand_mass, "entropy=", entropy, "margin=", margin)
    # print("[TeacherAlign] top1=", int(p.argmax().item()), "p1=", float(p.max().item()))

def build_primitive_tokens(K: int) -> Tuple[str, List[str]]:
    mask_tok = "<MASK>"
    prims = [f"<P{k:02d}>" for k in range(K)]
    return mask_tok, prims

def ensure_special_tokens(tokenizer, model, K: int):
    """
    Route A:
      1) add special tokens: <MASK>, <P00>..<P{K-1}>
      2) resize embedding
      3) initialize each <Pxx> from a real numeric token embedding (" 00"/"00"/fallback)
      4) assert each is SINGLE token
    """
    mask_tok, prims = build_primitive_tokens(K)
    add_list = [mask_tok] + prims

    # add to tokenizer
    special = {"additional_special_tokens": add_list}
    num_added = tokenizer.add_special_tokens(special)

    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

        # ✅ Initialize new token embeddings from existing numeric tokens
        with torch.no_grad():
            in_emb = model.get_input_embeddings().weight  # [V,D]
            out_layer = model.get_output_embeddings()
            out_emb = out_layer.weight if (out_layer is not None) else None

            # range of newly added ids (HF appends new tokens at the end)
            new_start = in_emb.shape[0] - num_added

            def _find_anchor_id(s: str) -> int:
                """
                Try to find a SINGLE-token id for s (prefer leading-space form).
                Fallback: average of digit token embeddings if needed.
                """
                # Prefer leading space: " 00"
                for cand in [s, s.lstrip(), " " + s.lstrip()]:
                    ids = tokenizer.encode(cand, add_special_tokens=False)
                    if len(ids) == 1:
                        return ids[0]
                return -1

            # Init <Pxx> from anchors "00".."K-1"
            for k in range(K):
                p_tok = f"<P{k:02d}>"
                p_id = tokenizer.encode(p_tok, add_special_tokens=False)[0]

                s = f"{k:02d}"
                anchor_id = _find_anchor_id(s)
                # print("[Anchor]", k, s, "->", anchor_id)

                if anchor_id >= 0:
                    in_emb[p_id].copy_(in_emb[anchor_id])
                    if out_emb is not None and out_emb.shape[0] == in_emb.shape[0]:
                        out_emb[p_id].copy_(out_emb[anchor_id])
                else:
                    # fallback: average embeddings of characters '0'..'9' tokens if available
                    digit_ids = []
                    for ch in list(s):
                        ids = tokenizer.encode(ch, add_special_tokens=False)
                        if len(ids) == 1:
                            digit_ids.append(ids[0])
                    if len(digit_ids) > 0:
                        mu_in = in_emb[digit_ids].mean(dim=0)
                        in_emb[p_id].copy_(mu_in)
                        if out_emb is not None and out_emb.shape[0] == in_emb.shape[0]:
                            mu_out = out_emb[digit_ids].mean(dim=0)
                            out_emb[p_id].copy_(mu_out)
                    else:
                        # last resort: keep random init (rare)
                        pass

            # Init <MASK> from a reasonable anchor token (space or "?" are common stable anchors)
            mask_id = tokenizer.encode(mask_tok, add_special_tokens=False)[0]
            anchor_mask = _find_anchor_id("?")
            if anchor_mask >= 0:
                in_emb[mask_id].copy_(in_emb[anchor_mask])
                if out_emb is not None and out_emb.shape[0] == in_emb.shape[0]:
                    out_emb[mask_id].copy_(out_emb[anchor_mask])

    # assert single-token
    bad = []
    for t in add_list:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids) != 1:
            bad.append((t, ids))
    if bad:
        raise ValueError(f"Some special tokens are not single-token: {bad}")

    # return ids for fast use
    mask_id = tokenizer.encode(mask_tok, add_special_tokens=False)[0]
    prim_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in prims]
    return mask_tok, prims, mask_id, torch.tensor(prim_ids, dtype=torch.long)

def pid_to_tokens_special(pid_row_cpu, K: int) -> List[str]:
    _, prims = build_primitive_tokens(K)
    return [prims[int(k)] for k in pid_row_cpu.tolist()]

def build_llm_prompt_special(tokens: List[str], s: int, e: int, K: int) -> str:
    mask_tok, prims = build_primitive_tokens(K)
    masked = tokens.copy()
    for i in range(s, e):
        masked[i] = mask_tok

    seq = " ".join(masked)
    vocab = " ".join(prims)

    # 末尾空格：保证 next-token 就是答案 token
    prompt = (
        "TASK: Predict the missing primitive token for the masked span.\n"
        f"K={K}\n"
        f"MASK_SPAN=[{s},{e})\n"
        "VOCAB: " + vocab + "\n"
        "RULES:\n"
        "- Output exactly ONE token from VOCAB.\n"
        "- No explanation.\n"
        "SEQUENCE:\n"
        + seq + "\n"
        "OUTPUT: "
    )
    return prompt if prompt.endswith(" ") else (prompt + " ")

# ============================================================
# Sanity check: candidate mass at next-token position
# ============================================================

# ============================================================
# Missing event sampler: define ONE event S=[s,e) from pid
# ============================================================
def sample_missing_event_span(pid_row_cpu: torch.Tensor, min_span: int = 2) -> Tuple[int, int, int]:
    """
    pid_row_cpu: [P] on CPU
    Return (s,e,k) where span is [s,e) and k is primitive id of the span.
    """
    P = int(pid_row_cpu.numel())
    pid_list = pid_row_cpu.tolist()

    spans = []
    s = 0
    while s < P:
        k = pid_list[s]
        e = s + 1
        while e < P and pid_list[e] == k:
            e += 1
        if (e - s) >= min_span:
            spans.append((s, e, k))
        s = e

    if len(spans) == 0:
        s = random.randint(0, max(0, P - min_span))
        e = min(P, s + min_span)
        k = pid_list[s]
        return s, e, k

    return random.choice(spans)


def span_to_patch_mask(P: int, s: int, e: int, device) -> torch.Tensor:
    m = torch.zeros(P, dtype=torch.bool, device=device)
    m[s:e] = True
    return m


# ============================================================
# Tokenization (task-level): pid -> ["00", "29", ...] and prompt
#   (Route A) NO special tokens needed
# ============================================================
MASK_TOKEN_TEXT = "[MASK]"


def pid_to_tokens(pid_row_cpu: torch.Tensor) -> List[str]:
    # two-digit strings
    return [f"{int(k):02d}" for k in pid_row_cpu.tolist()]


def build_llm_prompt(tokens: List[str], s: int, e: int, K: int = 32) -> str:
    masked = tokens.copy()
    for i in range(s, e):
        masked[i] = MASK_TOKEN_TEXT
    seq = " ".join(masked)

    vocab = " ".join([f"{k:02d}" for k in range(K)])

    # IMPORTANT:
    # - keep output on the SAME line; end with a space so next-token is the answer token
    prompt = (
        "TASK: Predict the missing primitive ID for the masked span.\n"
        f"K={K}\n"
        f"MASK_SPAN=[{s},{e})\n"
        "VOCAB: " + vocab + "\n"
        "RULES:\n"
        "- Output exactly ONE item from VOCAB.\n"
        "- Output must be exactly two digits.\n"
        "- No explanation. No extra characters.\n"
        "SEQUENCE:\n"
        + seq + "\n"
        "OUTPUT (one token from VOCAB): "
    )
    return prompt


# ============================================================
# Cache key: content-hash
# ============================================================
def make_content_key(dataset_key: str, pid_row_cpu: torch.Tensor, s: int, e: int, K: int, patch_len: int) -> str:
    pid_str = ",".join([f"{int(x):02d}" for x in pid_row_cpu.tolist()])
    raw = f"{dataset_key}|K{K}|pl{patch_len}|pid:{pid_str}|span:{s}-{e}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class TeacherCache:
    """
    JSON cache: key -> list length K (probabilities)
    """
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.db = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self.db = json.load(f)

    def get(self, key: str) -> Optional[torch.Tensor]:
        v = self.db.get(key, None)
        if v is None:
            return None
        return torch.tensor(v, dtype=torch.float32)

    def put(self, key: str, probs: torch.Tensor):
        self.db[key] = probs.detach().cpu().tolist()

    def save(self):
        if not self.cache_path:
            return
        d = os.path.dirname(self.cache_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.db, f)


# ============================================================
# Online LLM teacher (Route A): candidates are "00".."31"
#   - score_mode="next": gather next-token logits for candidate ids
#   - score_mode="cont": compute continuation logprob (for safety/debug)
# ============================================================
class OnlineLLMTeacher:
    """
    Route A (RECOMMENDED):
      - candidates are special tokens: <P00>..<P{K-1}>
      - mask token: <MASK>
      - scoring:
          * probs_next_batch: gather next-token logits for candidate ids (fast)
          * probs_cont_batch: continuation logprob (robust/debug; slower)
    """
    def __init__(
        self,
        name_or_path: str,
        K: int,
        device: str = "cuda",
        dtype: str = "fp16",
    ):
        self.K = int(K)
        self.device = device

        torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype, "auto")

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device.startswith("cuda") else None,
        ).eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ✅ Route A: add special tokens + resize + assert single-token
        mask_tok, prims, mask_id, prim_ids = ensure_special_tokens(self.tokenizer, self.model, self.K)

        self.mask_tok = mask_tok
        self.prims = prims                     # ["<P00>", ...]
        self.mask_id = int(mask_id)
        self.candidates = prims                # ✅ candidates are <Pxx>
        self.cand_ids_cpu = prim_ids.to("cpu") # ✅ [K] ids for <Pxx>

        # sanity: each candidate must be single token id and match tokenizer decode
        for t, tid in zip(self.candidates, self.cand_ids_cpu.tolist()):
            ids = self.tokenizer.encode(t, add_special_tokens=False)
            if len(ids) != 1 or ids[0] != tid:
                raise RuntimeError(f"[Sanity] candidate mismatch: {t} -> {ids}, stored={tid}")

    def _input_device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    @staticmethod
    def _ensure_trailing_space(prompts):
        return [p if (p and p.endswith(" ")) else (p + " ") for p in prompts]

    # -------------------------
    # Fast: next-token gather
    # -------------------------
    @torch.no_grad()
    def probs_next_batch(self, prompts, temperature: float = 1.0) -> torch.Tensor:
        prompts = self._ensure_trailing_space(prompts)
        tok = self.tokenizer
        dev = self._input_device()
        temp = max(float(temperature), 1e-6)

        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
        input_ids = enc["input_ids"].to(dev)
        attn_mask = enc["attention_mask"].to(dev)

        out = self.model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits  # [B,T,V]

        # last valid position per row
        last_idx = attn_mask.sum(dim=1) - 1
        last_logits = logits[torch.arange(logits.size(0), device=logits.device), last_idx, :]  # [B,V]

        cand_ids = self.cand_ids_cpu.to(last_logits.device)  # [K]

        logp = F.log_softmax(last_logits / temp, dim=-1)     # [B,V]
        cand_logp = logp.index_select(dim=-1, index=cand_ids)  # [B,K]

        # cand_mass in original softmax space:
        # cand_mass = sum_{c in candidates} softmax(last_logits)_c
        full_p = torch.softmax(last_logits / temp, dim=-1)  # [B,V]
        cand_mass = full_p.index_select(dim=-1, index=cand_ids).sum(dim=-1)  # [B]

        probs = torch.softmax(cand_logp, dim=-1)             # [B,K]
        return probs.detach().float().cpu(), cand_mass.detach().float().cpu()

    @torch.no_grad()
    def probs_next(self, prompt: str, temperature: float = 1.0):
        p, m = self.probs_next_batch([prompt], temperature=temperature)
        return p[0], m[0]

    # -------------------------
    # Robust/Debug: continuation logprob
    # -------------------------
    @torch.no_grad()
    def probs_cont_batch(self, prompts: list[str], temperature: float = 1.0) -> torch.Tensor:
        tok = self.tokenizer
        dev = self._input_device()
        temp = max(float(temperature), 1e-6)

        prompts = self._ensure_trailing_space(prompts)

        # Build B*K sequences: prompt + candidate
        full_texts = []
        for p in prompts:
            for c in self.candidates:   # ✅ <Pxx>
                full_texts.append(p + c)

        enc = tok(full_texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
        input_ids = enc["input_ids"].to(dev)      # [BK,T]
        attn_mask = enc["attention_mask"].to(dev) # [BK,T]

        out = self.model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits                       # [BK,T,V]
        logp_all = F.log_softmax(logits / temp, dim=-1)

        last_pos = attn_mask.sum(dim=1) - 1       # [BK]
        pred_pos = (last_pos - 1).clamp_min(0)

        target_ids = input_ids[torch.arange(input_ids.size(0), device=dev), last_pos]  # [BK]
        token_logp = logp_all[
            torch.arange(input_ids.size(0), device=dev), pred_pos, :
        ].gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)    # [BK]

        B = len(prompts)
        scores = token_logp.view(B, self.K).float().cpu()
        probs = torch.softmax(scores, dim=-1)
        return probs

    @torch.no_grad()
    def probs_cont(self, prompt: str, temperature: float = 1.0) -> torch.Tensor:
        return self.probs_cont_batch([prompt], temperature=temperature)[0]


# ============================================================
# Teacher provider: if/else switch between cache and online
# ============================================================
class TeacherProvider:
    """
    teacher_mode:
      - "cache": only cache (optional fallback online)
      - "online": always online (optional write cache)

    teacher_score_mode:
      - "next" : next-token gather (fast)   [recommended for Route A]
      - "cont" : continuation logprob (robust/debug)
    """
    def __init__(
        self,
        teacher_mode: str,
        K: int,
        dataset_key: str,
        patch_len: int,
        teacher_score_mode: str = "next",
        cache_path: str = "",
        llama_name: str = "",
        llm_device: str = "cuda",
        llm_dtype: str = "fp16",
        fallback_online_if_miss: bool = False,
        write_cache: bool = True,
        temperature: float = 1.0,
        debug: bool = False,
    ):
        self.teacher_mode = str(teacher_mode).lower()
        self.score_mode = str(teacher_score_mode).lower()
        assert self.score_mode in ["next", "cont"]

        self.K = int(K)
        self.dataset_key = str(dataset_key)
        self.patch_len = int(patch_len)
        self.temperature = float(temperature)

        self.cache = TeacherCache(cache_path) if cache_path else None
        self.online = None
        if self.teacher_mode == "online" or fallback_online_if_miss:
            if not llama_name:
                raise ValueError("Need llama_name for online teacher.")
            self.online = OnlineLLMTeacher(llama_name, self.K, device=llm_device, dtype=llm_dtype)

        self.fallback_online_if_miss = bool(fallback_online_if_miss)
        self.write_cache = bool(write_cache)
        self.debug = bool(debug)

    def _online_probs_batch(self, prompts: List[str]) -> torch.Tensor:
        if self.online is None:
            raise RuntimeError("online teacher is None but _online_probs_batch called.")
        if self.score_mode == "next":
            return self.online.probs_next_batch(prompts, temperature=self.temperature)
        else:
            return self.online.probs_cont_batch(prompts, temperature=self.temperature)

    def _online_probs(self, prompt: str) -> torch.Tensor:
        if self.online is None:
            raise RuntimeError("online teacher is None but _online_probs called.")
        if self.score_mode == "next":
            return self.online.probs_next(prompt, temperature=self.temperature)  # ✅ now exists
        else:
            return self.online.probs_cont(prompt, temperature=self.temperature)

    @torch.no_grad()
    def get_teacher_probs(self, pid_row_cpu: torch.Tensor, s: int, e: int) -> torch.Tensor:
        key = make_content_key(self.dataset_key, pid_row_cpu, s, e, self.K, self.patch_len)

        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        if self.teacher_mode == "cache" and not self.fallback_online_if_miss:
            return torch.full((self.K,), 1.0 / self.K, dtype=torch.float32)

        # tokens = pid_to_tokens(pid_row_cpu)
        # prompt = build_llm_prompt(tokens, s, e, K=self.K)
        tokens = pid_to_tokens_special(pid_row_cpu, K=self.K)
        prompt = build_llm_prompt_special(tokens, s, e, K=self.K)


        probs = self._online_probs(prompt).cpu()

        if self.cache is not None and self.write_cache:
            self.cache.put(key, probs)
        return probs

    @torch.no_grad()
    def get_teacher_probs_batch(self, pid_cpu: torch.Tensor, span_info: List[Tuple[int, int, int]]):
        """
        pid_cpu: [B,P] on CPU
        span_info: list of (s,e,k) length B
        return:
          probs: [B,K] on CPU float32  (candidate-normalized)
          cand_mass: [B]  on CPU float32 (probability mass of candidates in full softmax space)
        """
        B = pid_cpu.size(0)
        out = [None] * B
        mass_out = [None] * B
        miss_idx, miss_prompts, miss_keys = [], [], []

        for b in range(B):
            s, e, _ = span_info[b]
            key = make_content_key(self.dataset_key, pid_cpu[b], s, e, self.K, self.patch_len)

            # 1) cache hit
            if self.cache is not None:
                cached = self.cache.get(key)
                if cached is not None:
                    out[b] = cached
                    mass_out[b] = torch.tensor(1.0, dtype=torch.float32)  # ✅ cache: gate_mass=1
                    continue

            # 2) pure cache mode (no online fallback)
            if self.teacher_mode == "cache" and not self.fallback_online_if_miss:
                out[b] = torch.full((self.K,), 1.0 / self.K, dtype=torch.float32)
                mass_out[b] = torch.tensor(1.0, dtype=torch.float32)  # ✅ treat as reliable mass
                continue

            # 3) online miss → build prompt (Route A)
            tokens = pid_to_tokens_special(pid_cpu[b], K=self.K)
            prompt = build_llm_prompt_special(tokens, s, e, K=self.K)

            miss_idx.append(b)
            miss_prompts.append(prompt)
            miss_keys.append(key)

        # 4) query online for misses
        if len(miss_idx) > 0:
            if self.debug and self.online is not None:
                debug_teacher_alignment(self.online, miss_prompts[0])


            probs_bk, cand_mass_bm = self._online_probs_batch(miss_prompts)  # probs:[Bm,K], mass:[Bm]

            for j, b in enumerate(miss_idx):
                out[b] = probs_bk[j]
                mass_out[b] = cand_mass_bm[j]

                if self.cache is not None and self.write_cache:
                    self.cache.put(miss_keys[j], probs_bk[j])

        probs = torch.stack(out, dim=0)  # [B,K]
        cand_mass = torch.stack(mass_out, dim=0)  # [B]
        return probs, cand_mass

    def save_cache(self):
        if self.cache is not None:
            self.cache.save()


class KMeansPrimitiveAssigner(nn.Module):
    """
    P2: Assign each patch embedding to nearest KMeans center.
    centers: [K, D]
    """
    def __init__(self, centers: torch.Tensor):
        super().__init__()
        assert centers.dim() == 2, "centers must be [K,D]"
        self.register_buffer("centers", centers)  # [K,D]

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B,P,D]
        return pid: [B,P] long in [0..K-1]
        """
        B, P, D = z.shape
        c = self.centers.to(z.device)  # [K,D]
        # ||z-c||^2 = z^2 + c^2 - 2 z·c
        z2 = (z ** 2).sum(dim=-1, keepdim=True)          # [B,P,1]
        c2 = (c ** 2).sum(dim=-1).view(1, 1, -1)         # [1,1,K]
        dot = torch.matmul(z, c.t())                     # [B,P,K]
        dist2 = z2 + c2 - 2 * dot                        # [B,P,K]
        pid = dist2.argmin(dim=-1)
        return pid
def patch_rfft_amp(x_patch, patch_len):
    # x_patch: [B,P,patch_len*C] -> reshape -> rfft along time
    B, P, D = x_patch.shape
    C = D // patch_len
    xt = x_patch.view(B * P, patch_len, C).transpose(1, 2)  # [B*P,C,T]
    amp = torch.fft.rfft(xt.float(), dim=-1).abs()          # [B*P,C,F]
    return amp


# ============================================================
# Your original modules (Resampler, Decoder, PatchEmbedding...)
# (UNCHANGED except imports)
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
        self.num_patches = num_patches
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
        pred = self.pred_head(x)
        return pred


class AttnPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, max(1, dim // 2)),
            nn.GELU(),
            nn.Linear(max(1, dim // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.score(x)
        w = torch.softmax(w, dim=1)
        return (x * w).sum(dim=1)


class PatchEmbeddingConv(nn.Module):
    def __init__(self, seq_len: int, patch_len: int, in_channels: int, embed_dim: int):
        super().__init__()
        assert seq_len % patch_len == 0
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.in_channels = in_channels
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

    def forward(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor] = None, return_pre_pos=False) -> torch.Tensor:
        B, L, C = x.shape
        assert L == self.seq_len, f"expect L={self.seq_len}, got {L}"

        h = self.stem(x.transpose(1, 2))  # [B,D,L]
        h = h.transpose(1, 2).contiguous()             # [B,L,D]

        h = torch.reshape(h, (B, self.num_patches, self.patch_len * self.embed_dim))
        h = self.proj(h)
        h = self.norm(h)
        pre_pos = h  # [B,P,D] 位置无关
        h = h + self.pos_embed

        if patch_mask is not None:
            mask_tokens = self.mask_token.expand(B, self.num_patches, self.embed_dim)
            w = patch_mask.unsqueeze(-1).type_as(h)
            h = h * (1 - w) + mask_tokens * w
        return (h, pre_pos) if return_pre_pos else h

# ============================================================
# Model with LLM-guided KD (StageC) + P2(KMeans)
# ============================================================
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        # -------- dataset cfg load --------
        self.dataset_key = str(getattr(args, "dataset_key", getattr(args, "data", "mhealth"))).lower()
        self.ds_cfg: Dict[str, Any] = {}
        if hasattr(args, "ds_cfg") and isinstance(args.ds_cfg, dict):
            self.ds_cfg = args.ds_cfg
        else:
            ts_yaml = getattr(args, "ts_backbone_yaml", None)
            if ts_yaml is not None:
                if yaml is None:
                    raise ImportError("pyyaml not installed but ts_backbone_yaml is set.")
                if not os.path.exists(ts_yaml):
                    raise FileNotFoundError(ts_yaml)
                with open(ts_yaml, "r", encoding="utf-8") as f:
                    cfg_all = yaml.safe_load(f)
                if self.dataset_key not in cfg_all:
                    raise KeyError(f"{self.dataset_key} not in {ts_yaml}")
                self.ds_cfg = cfg_all[self.dataset_key]

        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 15)))
        self.num_class = int(self.ds_cfg.get("num_labels", getattr(args, "num_class", 12)))
        self.stage = int(getattr(args, "stage", 1))

        self.seq_len = int(getattr(args, "seq_len", 200))
        self.patch_len = int(getattr(args, "patch_len", 20))
        assert self.seq_len % self.patch_len == 0
        self.P = self.seq_len // self.patch_len

        # MAE random mask (warmup 使用)
        self.mask_rate = float(getattr(args, "mask_rate", 0.5))

        self.resampler_dim = int(getattr(args, "resampler_dim", 256))
        self.num_latents = int(getattr(args, "num_latents", min(self.P, 16)))

        # ---- backbone modules ----
        self.patch_embed = PatchEmbeddingConv(
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            in_channels=self.C,
            embed_dim=self.resampler_dim
        )

        self.resampler = DeepResampler(
            hidden_size=self.resampler_dim,
            num_latents=self.num_latents,
            depth=int(getattr(args, "resampler_depth", 4)),
            num_heads=int(getattr(args, "num_heads", 8))
        )

        self.mae_decoder = MAEDecoder(
            hidden_size=self.resampler_dim,
            num_patches=self.P,
            patch_dim=self.patch_len * self.C,
            depth=int(getattr(args, "dec_depth", 2))
        )

        self.pool = AttnPool(self.resampler_dim)
        self.cls_head = nn.Linear(self.resampler_dim, self.num_class)

        self.norm_eps = float(getattr(args, "norm_eps", 1e-5))
        self.loss_fp32 = bool(getattr(args, "loss_fp32", True))

        # ==========================
        # P2(KMeans) + LLM teacher (KD)
        # ==========================
        self.K = int(getattr(args, "primitive_K", 32))
        self.warmup_only = int(getattr(args, "warmup_only", 1))

        # KD 相关超参（forward 里会用到，必须永远存在）
        self.min_span = int(getattr(args, "min_span", 2))
        self.lambda_kd = float(getattr(args, "lambda_kd", 0.2))

        # teacher 配置字段（不管 warmup_only 与否都先存好，避免 enable_kd() 时缺字段）
        self.teacher_mode = str(getattr(args, "teacher_mode", "cache")).lower()
        self.teacher_cache_path = str(getattr(args, "teacher_cache_path", ""))
        self.fallback_online_if_miss = bool(getattr(args, "fallback_online_if_miss", False))
        self.write_cache = bool(getattr(args, "write_cache", True))
        self.teacher_temp = float(getattr(args, "teacher_temp", 1.0))

        self.llama_name = str(getattr(args, "llama_name", ""))
        self.llm_device = str(getattr(args, "llm_device", "cuda"))
        self.llm_dtype = str(getattr(args, "llm_dtype", "fp16"))

        # 运行期对象（warmup 可以为空；KD 开启后必须非空）
        self.primitive_assigner: Optional[KMeansPrimitiveAssigner] = None
        self.missing_head: Optional[nn.Module] = None
        self.teacher: Optional[TeacherProvider] = None

        # ---- 如果一开始就是 KD 模式（warmup_only=0），则要求给 centers 并初始化 ----
        if self.warmup_only != 1:
            centers_path = getattr(args, "kmeans_centers_path", None)
            if centers_path is None:
                raise ValueError("warmup_only=False requires args.kmeans_centers_path (torch-saved [K,D]).")
            centers = torch.load(centers_path, map_location="cpu")

            self.set_kmeans_centers(centers)                 # -> primitive_assigner
            # enable_kd() 里：
            self.missing_head = nn.Sequential(
                nn.LayerNorm(self.resampler_dim * 4),
                nn.Linear(self.resampler_dim* 4, self.resampler_dim*2),
                nn.GELU(),
                nn.Linear(self.resampler_dim*2, self.K),
            )

            self.teacher = TeacherProvider(
                teacher_mode=self.teacher_mode,
                K=self.K,
                dataset_key=self.dataset_key,
                patch_len=self.patch_len,
                cache_path=self.teacher_cache_path,
                llama_name=self.llama_name,
                llm_device=self.llm_device,
                llm_dtype=self.llm_dtype,
                fallback_online_if_miss=self.fallback_online_if_miss,
                write_cache=self.write_cache,
                temperature=self.teacher_temp,
            )

    # -------------------------
    # helper
    # -------------------------
    def _align_seq_len(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        if L != self.seq_len:
            if L > self.seq_len:
                x = x[:, -self.seq_len:, :]
            else:
                pad = self.seq_len - L
                x = torch.cat([torch.zeros(B, pad, C, device=x.device, dtype=x.dtype), x], dim=1)
        return x

    # -------------------------
    # ✅ 热插拔：注入 centers
    # -------------------------
    @torch.no_grad()
    def set_kmeans_centers(self, centers: torch.Tensor):
        if not torch.is_tensor(centers):
            centers = torch.as_tensor(centers)
        centers = centers.detach().to("cpu", dtype=torch.float32)

        if centers.dim() != 2:
            raise ValueError(f"centers must be [K,D], got {tuple(centers.shape)}")
        if centers.shape != (self.K, self.resampler_dim):
            raise ValueError(f"centers must be [K,D]=[{self.K},{self.resampler_dim}], got {tuple(centers.shape)}")

        self.primitive_assigner = KMeansPrimitiveAssigner(centers)

    # -------------------------
    # ✅ 热切换：开启 KD（在 Stage1 warmup 后调用）
    # -------------------------
    def enable_kd(self):
        # 切换开关
        self.warmup_only = 0

        # 必须已有 centers
        if self.primitive_assigner is None:
            raise RuntimeError("enable_kd() requires primitive_assigner. Call set_kmeans_centers() first.")

        # 缺了就补齐
        if self.missing_head is None:
            # enable_kd() 里：
            self.missing_head = nn.Sequential(
                nn.LayerNorm(self.resampler_dim * 4),
                nn.Linear(self.resampler_dim * 4, self.resampler_dim * 2),
                nn.GELU(),
                nn.Linear(self.resampler_dim * 2, self.K),
            ).to(self.device)

        if self.teacher is None:
            # 如果要在线 teacher，必须给 llama_name
            if (self.teacher_mode == "online" or self.fallback_online_if_miss) and (not self.llama_name):
                raise ValueError("KD teacher needs args.llama_name when teacher_mode='online' or fallback_online_if_miss=True.")

            self.teacher = TeacherProvider(
                teacher_mode=self.teacher_mode,
                K=self.K,
                dataset_key=self.dataset_key,
                patch_len=self.patch_len,
                cache_path=self.teacher_cache_path,
                llama_name=self.llama_name,
                llm_device=self.llm_device,
                llm_dtype=self.llm_dtype,
                fallback_online_if_miss=self.fallback_online_if_miss,
                write_cache=self.write_cache,
                temperature=self.teacher_temp,
            )


        # enable_kd() 末尾：
        if bool(getattr(self.args, "freeze_patch_embed_after_kd", True)):
            self._freeze_to_prepos()

    def _freeze_to_prepos(self):
        # 冻结 patch_embed 的表征路径（stem/proj/norm/pos/mask 都冻掉最稳）
        for n, p in self.patch_embed.named_parameters():
            p.requires_grad = False

    def forward(self, batch_x, padding_mask=None, mode: Optional[str] = None, labels=None):
        if mode is None:
            mode = "pretrain" if self.stage == 1 else "classify"

        if not torch.is_tensor(batch_x):
            batch_x = torch.as_tensor(batch_x)
        x = self._align_seq_len(batch_x)

        # instance norm
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True).clamp_min(self.norm_eps)
        x_norm = (x - mu) / sigma

        # -------------------------
        # Stage2: classify
        # -------------------------
        if mode == "classify":
            z = self.patch_embed(x_norm)  # [B,P,D]
            lat = self.resampler(z)       # [B,M,D]
            pooled = lat.mean(dim=1)
            logits = self.cls_head(pooled)
            return logits

        # -------------------------
        # Stage1: pretrain (MAE + KD)
        # -------------------------
        # -------------------------
        # Stage1: pretrain
        #   - warmup_only: pure MAE with random patch mask (no pid / no teacher)
        #   - else: MAE + KD (pid-span event aligned)
        # -------------------------
        B = x_norm.size(0)

        # ---------- (0) warmup-only: pure MAE ----------
        if self.warmup_only:
            # random patch mask (same as your original MAE)
            patch_mask = (torch.rand(B, self.P, device=x_norm.device) < self.mask_rate)
            if self.P >= 2:
                all_masked = patch_mask.all(dim=1)
                none_masked = (~patch_mask).all(dim=1)
                if all_masked.any():
                    patch_mask[
                        all_masked, torch.randint(0, self.P, (all_masked.sum().item(),), device=x_norm.device)] = False
                if none_masked.any():
                    patch_mask[
                        none_masked, torch.randint(0, self.P, (none_masked.sum().item(),), device=x_norm.device)] = True

            z = self.patch_embed(x_norm, patch_mask=patch_mask)  # masked tokens
            lat = self.resampler(z)
            pred = self.mae_decoder(lat)

            target = x_norm.view(B, self.P, self.patch_len * x_norm.size(2))
            if self.loss_fp32:
                pred = pred.float()
                target = target.float()

            loss_patch = (pred - target).pow(2).mean(dim=-1)
            loss_recon = (loss_patch * patch_mask.float()).sum() / (patch_mask.float().sum() + 1e-6)

            return loss_recon, patch_mask, {
                "P": float(self.P),
                "mask_rate": float(patch_mask.float().mean().item()),
                "loss_recon": float(loss_recon.item()),
                "loss": float(loss_recon.item()),
            }

        # ---------- (1) KD mode requires centers/teacher ----------
        if self.primitive_assigner is None or self.missing_head is None or self.teacher is None:
            raise RuntimeError("KD mode requires primitive_assigner/missing_head/teacher. Check warmup_only and paths.")

        # (A) primitive ids from unmasked embeddings
        # z_nomask = self.patch_embed(x_norm, patch_mask=None)  # [B,P,D]
        z_nomask, z_prepos = self.patch_embed(x_norm, patch_mask=None, return_pre_pos=True)
        # pid = self.primitive_assigner(z_nomask)  # [B,P]
        pid = self.primitive_assigner(z_prepos)  # ✅ position-invariant primitive
        # (B) define ONE missing event S per sample: span [s,e)
        patch_mask = torch.zeros(B, self.P, dtype=torch.bool, device=x_norm.device)
        span_info = []
        for b in range(B):
            s, e, k = sample_missing_event_span(pid[b].detach().cpu(), min_span=self.min_span)
            span_info.append((s, e, k))
            patch_mask[b] = span_to_patch_mask(self.P, s, e, device=x_norm.device)

        # (C) student MAE path: same event S -> numeric mask token
        z = self.patch_embed(x_norm, patch_mask=patch_mask)
        lat = self.resampler(z)
        pred = self.mae_decoder(lat)

        target = x_norm.view(B, self.P, self.patch_len * x_norm.size(2))
        if self.loss_fp32:
            pred = pred.float()
            target = target.float()

        loss_patch = (pred - target).pow(2).mean(dim=-1)
        loss_recon = (loss_patch * patch_mask.float()).sum() / (patch_mask.float().sum() + 1e-6)






        # (D) student missing primitive prediction (event-aware):
        # Use boundary context from *unmasked* patch embeddings z_nomask for stability.
        # ctx_list = []
        # for b in range(B):
        #     s, e, _ = span_info[b]
        #     # left boundary
        #     if s - 1 >= 0:
        #         left = z_prepos[b, s - 1]
        #     else:
        #         left = z_prepos[b, 0]
        #     # right boundary
        #     if e < self.P:
        #         right = z_prepos[b, e]
        #     else:
        #         right = z_prepos[b, self.P - 1]
        #     ctx = 0.5 * (left + right)   # [D]
        #     ctx_list.append(ctx)
        # ctx = torch.stack(ctx_list, dim=0)               # [B,D]
        #
        # logits_enc = self.missing_head(ctx)              # [B,K]
        #

        def _safe_idx(i, P):
            return max(0, min(P - 1, int(i)))

        ctx_list = []
        for b in range(B):
            s, e, _ = span_info[b]
            idxs = [_safe_idx(s - 2, self.P), _safe_idx(s - 1, self.P),
                    _safe_idx(e, self.P), _safe_idx(e + 1, self.P)]
            feats = [z_prepos[b, j] for j in idxs]  # 4 x [D]
            ctx_b = torch.cat(feats, dim=-1)  # [4D]
            ctx_list.append(ctx_b)

        ctx = torch.stack(ctx_list, dim=0)  # [B,4D]
        logits_enc = self.missing_head(ctx)  # [B,K]
        # logp_enc = F.log_softmax(logits_enc, dim=-1)

        logp_enc = F.log_softmax(logits_enc, dim=-1)             # [B,K] log-prob

        # (E) teacher: same event S -> prompt [MASK] -> p_llm
        p_llm_list = []

        pid_cpu = pid.detach().cpu()  # [B,P] 一次性搬到 CPU，避免 B 次同步
        p_llm_cpu, cand_mass_cpu = self.teacher.get_teacher_probs_batch(pid_cpu, span_info)  # [B,K], [B] on CPU

        p_llm = p_llm_cpu.to(x_norm.device)  # [B,K]
        cand_mass = cand_mass_cpu.to(x_norm.device)  # [B]

        # candidate-normalized probs (should already sum to 1, but keep for safety)
        p_llm = (p_llm / (p_llm.sum(dim=-1, keepdim=True) + 1e-9)).clamp_min(1e-9)

        # debug
        # print("[KD] p_llm shape:", tuple(p_llm.shape), "sum0:", float(p_llm[0].sum().item()))
        # print("[KD] conf mean:", float(p_llm.max(dim=-1).values.mean().item()))
        # print("[KD] cand_mass mean:", float(cand_mass.mean().item()))

        if True:
            # 1) prompt token check: must contain <MASK> and <Pxx>
            # (only print once per run if you want; here just minimal)
            self.teacher.debug = False  # optional: avoid too much debug spam




        # logp_enc: [B,K]  (log_softmax)
        # p_llm   : [B,K]  (prob, normalized, clamp_min)

        # conf = p_llm.max(dim=-1).values  # [B]
        # tau = float(getattr(self.args, "kd_tau", 0.13))
        # gate = (conf >= tau).float()  # [B]
        #
        # kl_per = F.kl_div(logp_enc, p_llm, reduction="none").sum(dim=-1)  # [B]
        # loss_kd = (gate * kl_per).sum() / (gate.sum() + 1e-6)
        # p_llm: [B,K] prob, logp_enc: [B,K] log-prob
        # conf gate
        conf = p_llm.max(dim=-1).values
        tau = float(getattr(self.args, "kd_tau", 0.13))
        t_gate = float(getattr(self.args, "kd_gate_temp", 0.05))
        gate_conf = torch.sigmoid((conf - tau) / max(t_gate, 1e-6))

        conf = p_llm.max(dim=-1).values  # [B]

        # target keep rate: only top-r samples get strong KD
        keep_rate = float(getattr(self.args, "kd_keep_rate", 0.6))  # 0.5~0.7 推荐
        # tau = (1-keep_rate) 分位数，例如 keep_rate=0.6 -> tau=conf的40分位
        tau = torch.quantile(conf.detach(), q=max(0.0, min(1.0, 1.0 - keep_rate))).item()

        t_gate = float(getattr(self.args, "kd_gate_temp", 0.05))
        gate_conf = torch.sigmoid((conf - tau) / max(t_gate, 1e-6))

        # mass gate (NEW)  —— 需要你把 cand_mass 从 teacher 拿回来
        mass = cand_mass.to(x_norm.device)  # [B]
        mass_tau = float(getattr(self.args, "kd_mass_tau", 0.05))  # 建议 0.02~0.10
        mass_temp = float(getattr(self.args, "kd_mass_temp", 0.02))  # 建议 0.01~0.05
        gate_mass = torch.sigmoid((mass - mass_tau) / max(mass_temp, 1e-6))

        gate = gate_conf * gate_mass

        kl_per = F.kl_div(logp_enc, p_llm, reduction="none").sum(dim=-1)  # [B]
        loss_kd = (gate * kl_per).sum() / (gate.sum() + 1e-6)

        # print("[KD] gate_conf mean:", float(gate_conf.mean().item()))
        # print("[KD] gate_mass mean:", float(gate_mass.mean().item()))
        # print("[KD] gate mean:", float(gate.mean().item()))

        # loss_kd = F.kl_div(logp_enc, p_llm, reduction="batchmean")

        loss = loss_recon + self.lambda_kd * loss_kd

        # (optional) periodically save cache (write-through)
        if bool(getattr(self.args, "save_teacher_cache_each_forward", False)):
            self.teacher.save_cache()

        return loss, patch_mask, {
            "P": float(self.P),
            "mask_rate": float(patch_mask.float().mean().item()),
            "loss_recon": float(loss_recon.item()),
            "loss_kd": float(loss_kd.item()),
            "loss": float(loss.item()),
            "llm_conf_mean": float(conf.mean().item()),
            "llm_conf_p50": float(conf.median().item()),
            "kd_gate_rate": float(gate.mean().item()),
        }

    def save_wrapper(self, path: str):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_wrapper(self, path: str, map_location="cpu"):
        sd = torch.load(path, map_location=map_location)
        self.load_state_dict(sd, strict=False)


# ============================================================
# Minimal usage notes (you set args):
# ============================================================
"""
args required for P2 + teacher:

# --- primitive P2 ---
args.primitive_K = 32
args.kmeans_centers_path = "/path/to/kmeans_centers_K32_D256.pt"   # torch.save(tensor[K,D])

# --- teacher if/else ---
args.teacher_mode = "cache"   # or "online"

# --- cache mode ---
args.teacher_cache_path = "/path/to/teacher_cache.json"
args.fallback_online_if_miss = False   # True if you want online fallback on cache miss

# --- online mode ---
args.llama_name = "/path/to/local/llama"   # or HF repo id
args.llm_device = "cuda"
args.llm_dtype = "auto"   # "fp16" or "bf16"
args.write_cache = True
args.teacher_temp = 1.0

# KD strength
args.lambda_kd = 0.5
args.min_span = 2
"""

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
    parser.add_argument('--patch_len', type=int, default=50, help='patch length')
    parser.add_argument('--stride', type=int, default=50, help='stride')

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
    parser.add_argument('--ts_backbone_yaml', type=str, default=r'D:\fuy\MyCode\SensorLLMLib_version2\configs\ts_backbone.yaml', help='ts_backbone_yaml')
    parser.add_argument('--log_dir', type=str, default='./logs', help='log_dir')
    parser.add_argument('--dataset_key', type=str, default='mhealth', help='dataset_key')
    parser.add_argument('--stage', type=int, default=1, help='stage')
    parser.add_argument('--llama_name', type=str, default=r"D:\fuy\MyCode\SensorLLM\Llama-3.2-1B", help='stage')
    parser.add_argument('--two_stage', type=int, default=1, help='two_stage')
    parser.add_argument('--freeze_llm', type=int, default=0, help='freeze_llm')
    parser.add_argument('--trainable_modules', type=str,
                        default="sensor_patch_proj,channel_id,patch_pos,mask_embed,recon_head,cls_head,pool_query,pool_attn",
                        help='trainable_modules')
    parser.add_argument('--run_id', type=str, default="202501220_0956", help='trainable_modules')
    parser.add_argument('--test_subjects', type=str, default="202501220_0956", help='test_subjects')
    parser.add_argument('--pamap_variant', type=str, default="202501220_0956", help='pamap_variant')

    # datasets motion
    parser.add_argument('--test_users', type=str, default="19,20,21,22,23,24", help='test_users')
    parser.add_argument('--val_users', type=str, default="13,14,15,16,17,18", help='val_users')
    parser.add_argument('--motionsense_feature_set', type=str, default="A12", help='motionsense_feature_set')
    parser.add_argument('--motionsense_combine_grav_acc', type=int, default=0, help='motionsense_combine_grav_acc')
    parser.add_argument('--motionsense_norm', type=str, default='none', help='motionsense_norm')

    parser.add_argument('--hhar_tol', type=float, default=0.05, help='hhar_tol')
    parser.add_argument('--hhar_align_on', type=str, default="Arrival_Time", help='hhar_align_on')
    parser.add_argument('--hhar_use_cache', type=int, default=1, help='hhar_use_cache')
    parser.add_argument('--hhar_norm', type=str, default='none', help='hhar_norm')
    parser.add_argument('--wisdm_norm', type=str, default='none', help='wisdm_norm')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='val_ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test_ratio')

    parser.add_argument('--warmup_only', type=int, default=1, help='warmup_only')
    parser.add_argument('--kmeans_trigger_epoch', type=int, default=1, help='kmeans_trigger_epoch')
    parser.add_argument('--build_kmeans_centers', type=int, default=1, help='build_kmeans_centers')
    parser.add_argument('--kmeans_centers_path', type=str, default=r"D:\kmeans_centers_K32.pt", help='kmeans_centers_path')
    parser.add_argument('--teacher_mode', type=str, default="online", help='teacher_mode')


    args = parser.parse_args()
    return args
if __name__ == '__main__':
    z = torch.randn(16,2,256)
    c = torch.randn(32,256)
    end = torch.matmul(z, c.t())



    configs = get_configs()
    LLAMA_NAME = r"D:\fuy\MyCode/Llama-3.2-1B"
    ALIGN_W_MAX = 200
    SEQ_LEN = 256
    PATCH_LEN = 64
    BATCH_SIZE = 32
    LR = 0.001
    EPOCHS = 8
    model_name = 'SensorLLMFuy_test_withllm_mae'
    PRETRAIN_trainable_modules = "patch_embed,resampler,mae_decoder"
    TRAIN_trainable_modules = "patch_embed,resampler,llm_proj,cls_head"
    DATA_ROOT=r"D:\fuy\MyCode\SensorLLMLib\datasets\heterogeneity+activity+recognition\Activity recognition exp\Activity recognition exp"   # <- 改成你的路径(含 Phones_*.csv / Watch_*.csv)
    DATA_KEY="hhar"
    DATA_NAME="HHAR_1user"

    configs.task_name = 'classification'
    configs.is_training = 1
    configs.root_path = DATA_ROOT
    configs.model_id = 'HHAR_1user'
    configs.run_id = "$RUN_ID"
    configs.datasets =DATA_NAME
    configs.model = model_name
    configs.data =DATA_NAME
    configs.dataset_key =DATA_KEY
    configs.seq_len =SEQ_LEN
    configs.patch_len =PATCH_LEN
    configs.stride =PATCH_LEN
    configs.stage = 1
    configs.batch_size =BATCH_SIZE
    configs.trainable_modules =PRETRAIN_trainable_modules
    configs.llama_name =LLAMA_NAME
    configs.learning_rate =LR
    configs.train_epochs=EPOCHS
    configs.num_workers = 10
    configs.hhar_tol = 0.05
    configs.hhar_align_on = 'Arrival_Time'
    configs.hhar_use_cache = 1
    configs.hhar_norm = 'none'

    configs.task_name = "classification"
    configs.is_training = 1
    configs.root_path = r"D:\fuy\MyCode\SensorLLMLib\datasets\motion-sense-master\motion-sense-master\data"
    configs.model_id = "MotionSense"
    configs.run_id = "$RUN_ID"
    configs.datasets = "MotionSense"
    configs.model = "SensorLLMFuy_test_withllm_mae"
    configs.data = "MotionSense"
    configs.dataset_key = "motionsense"
    configs.seq_len = 256
    configs.patch_len = 64
    configs.stride = 64
    configs.stage = 1
    configs.batch_size = 64
    configs.llama_name = r"D:\fuy\MyCode\SensorLLM\Llama-3.2-1B"
    configs.learning_rate = 0.001
    configs.train_epochs = 1
    configs.num_workers = 0
    configs.warmup_only = 0
    configs.kmeans_trigger_epoch = 1
    configs.build_kmeans_centers = 1
    configs.teacher_mode = 'online'
    configs.test_users = "19,20,21,22,23,24"
    configs.val_users = "13,14,15,16,17,18"
    configs.motionsense_feature_set = "A12"
    configs.motionsense_combine_grav_acc = 0
    configs.motionsense_norm = "none"
    configs.device = "cuda:0"

    model = Model(configs)

    x = torch.randn(BATCH_SIZE, configs.seq_len, 12)

    c = model(x,None,'pretrain',None)
    end = 'emd '