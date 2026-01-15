# code2_teacher_infill_minimal.py
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass
class TeacherOut:
    probs: torch.Tensor  # [B,K]
    ok: torch.Tensor     # [B]

def build_infill_prompt(ids_1d: List[int], s: int, e: int) -> str:
    # single <MASK> token for the missing span
    toks = []
    for i, k in enumerate(ids_1d):
        if s <= i < e:
            if (len(toks) == 0) or (toks[-1] != "<MASK>"):
                toks.append("<MASK>")
        else:
            toks.append(str(int(k)))
    seq = " ".join(toks)
    prompt = (
        "Task: The following is a discrete motion primitive sequence with one missing span.\n"
        f"Sequence: {seq}\n"
        "Infer the missing primitive category for <MASK>.\n"
        "Answer with exactly one number token.\n"
        "Answer:"
    )
    return prompt

class InfillLLMTeacher:
    def __init__(self, llm_path: str, K: int, device: torch.device):
        self.K = K
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path).to(device)
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad = False

        # ---- build stable vocab map: require single-token encoding for " 0".." K-1"
        ids = []
        for i in range(K):
            s = " " + str(i)  # leading space often helps make it a single token in BPE tokenizers
            enc = self.tok.encode(s, add_special_tokens=False)
            if len(enc) != 1:
                raise RuntimeError(f'Primitive label "{s}" is not single-token (got {enc}). '
                                   f"Change label format or implement multi-token extraction.")
            ids.append(enc[0])
        self.label_token_ids = torch.tensor(ids, device=device, dtype=torch.long)

    @torch.no_grad()
    def __call__(self, prompts: List[str], max_length: int = 512) -> TeacherOut:
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
        out = self.llm(**enc)
        logits = out.logits  # [B,T,V]

        attn = enc["attention_mask"]
        last_idx = attn.sum(dim=1) - 1
        last_idx = last_idx.clamp_min(0)
        B = logits.size(0)
        last = logits[torch.arange(B, device=logits.device), last_idx, :]  # [B,V]

        sel = last.index_select(-1, self.label_token_ids)  # [B,K]
        probs = torch.softmax(sel, dim=-1)
        ok = torch.ones(B, device=probs.device, dtype=torch.bool)
        return TeacherOut(probs=probs, ok=ok)

def pick_span(P: int, span_len: int, device) -> Tuple[int, int]:
    span_len = max(1, min(P, span_len))
    s = int(torch.randint(0, max(1, P - span_len + 1), (1,), device=device).item())
    return s, s + span_len

if __name__ == "__main__":
    # demo (replace with your primitive_ids [B,P] loaded from cache)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    llm_path = r"D:\fuy\MyCode\Llama-3.2-1B"
    K = 32
    teacher = InfillLLMTeacher(llm_path, K=K, device=device)

    B, P = 2, 10
    primitive_ids = torch.randint(0, K, (B, P))

    span_len = 2
    prompts = []
    for b in range(B):
        s, e = pick_span(P, span_len, device=primitive_ids.device)
        prompts.append(build_infill_prompt(primitive_ids[b].tolist(), s, e))

    out = teacher(prompts)
    print(out.probs.shape, out.probs[0, :5])
