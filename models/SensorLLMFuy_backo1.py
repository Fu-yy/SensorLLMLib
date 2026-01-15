import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class SensorPromptLlama(nn.Module):
    """
    LLaMA wrapper: build channel-wise prompt with <TS_i> placeholders,
    replace placeholder embeddings with sensor embeddings, then run LLaMA.

    Input:
      batch_x: [B, L, C]  (e.g., [16, 96, 15])
      padding_mask: [B, L] (all-ones for fixed windows)
    Output:
      llm_outputs: model outputs (logits/hidden_states depending on flags)
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # ---- config ----
        self.C = int(getattr(args, "enc_in", 15))        # number of channels
        self.d_model = int(getattr(args, "d_model", 256))  # sensor embedding dim -> should match llama hidden size
        self.mask_rate = float(getattr(args, "mask_rate", 0.3))

        self.llama_name = getattr(args, "llama_name", "meta-llama/Llama-2-7b-hf")
        self.max_prompt_len = int(getattr(args, "max_prompt_len", 512))

        # ---- tokenizer + model ----
        self.tokenizer = AutoTokenizer.from_pretrained(self.llama_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            # LLaMA often has no pad token; set to eos to make attention_mask easy.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llama_name,
            torch_dtype=getattr(torch, getattr(args, "llm_dtype", "float16"), torch.float16),
        )

        # ---- add <TS_i> special tokens ----
        self.ts_tokens = [f"<TS_{i}>" for i in range(self.C)]
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": self.ts_tokens})
        if added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))

        # ---- sensor encoder (you can replace with your own) ----
        # Here we do a simple example: per-channel temporal pooling + linear -> [B,C,H]
        llama_hidden = self.llm.config.hidden_size
        self.sensor_proj = nn.Linear(1, llama_hidden)  # will be applied after pooling per channel

        # optional: channel id embedding to help distinguish channels
        self.channel_id = nn.Embedding(self.C, llama_hidden)

        # ---- optional: turn on hidden_states for downstream heads ----
        self.output_hidden_states = bool(getattr(args, "output_hidden_states", True))

    # -------------------------
    # 1) masking (simple example)
    # -------------------------
    def _random_mask(self, x: torch.Tensor):
        """
        x: [B,L,C]
        return:
          x_masked: [B,L,C] (masked values set to 0)
          mask: [B,L,C] bool (True where masked)
        """
        B, L, C = x.shape
        device = x.device
        mask = (torch.rand(B, L, C, device=device) < self.mask_rate)
        x_masked = x.clone()
        x_masked[mask] = 0.0
        return x_masked, mask

    # -------------------------
    # 2) sensor -> [B,C,H]
    # -------------------------
    def _encode_sensor(self, x_masked: torch.Tensor):
        """
        x_masked: [B,L,C]
        output: sensor_embeds [B,C,H]
        A simple baseline: mean-pool each channel over time, then project.
        """
        B, L, C = x_masked.shape
        # mean over time per channel: [B,C]
        pooled = x_masked.mean(dim=1)  # [B,C]
        # project each channel scalar -> H
        pooled = pooled.unsqueeze(-1)  # [B,C,1]
        sensor_embeds = self.sensor_proj(pooled)  # [B,C,H]

        # add channel identity embedding
        ch = torch.arange(C, device=x_masked.device).unsqueeze(0).expand(B, C)  # [B,C]
        sensor_embeds = sensor_embeds + self.channel_id(ch)  # [B,C,H]
        return sensor_embeds

    # -------------------------
    # 3) build channel-wise prompt
    # -------------------------
    def _build_prompt_text(self):
        """
        One prompt per sample (same template). We will batch-tokenize it.
        """
        lines = []
        lines.append("You are a sensor time-series reconstruction assistant.")
        lines.append(f"The input contains {self.C} channels from an IMU dataset (mHealth).")
        lines.append("Each channel is represented by a special token placeholder.")
        lines.append("Reconstruct the missing values for the masked time series.")
        lines.append("")
        lines.append("Channels:")
        for i in range(self.C):
            # Keep it short so prompt length stays stable
            lines.append(f"ch{i}: <TS_{i}>")
        lines.append("")
        lines.append("Return the reconstructed values for the masked positions.")
        return "\n".join(lines)

    @torch.no_grad()
    def _tokenize_prompt(self, B: int, device):
        """
        Returns:
          input_ids: [B,T]
          attn_mask: [B,T]
        """
        prompt = self._build_prompt_text()
        enc = self.tokenizer(
            [prompt] * B,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_len,
            return_tensors="pt",
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    # -------------------------
    # 4) replace <TS_i> embeddings in prompt embeddings
    # -------------------------
    def _inject_sensor_embeds(self, input_ids: torch.Tensor, sensor_embeds: torch.Tensor):
        """
        input_ids: [B,T]
        sensor_embeds: [B,C,H]
        return:
          inputs_embeds: [B,T,H] with placeholders replaced
        """
        B, T = input_ids.shape
        H = self.llm.config.hidden_size

        # base token embeddings
        embed_layer = self.llm.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)  # [B,T,H]

        # replace each <TS_i> position
        for i in range(self.C):
            ts_id = self.tokenizer.convert_tokens_to_ids(f"<TS_{i}>")
            pos = (input_ids == ts_id)  # [B,T] bool

            # safety: ensure each sample has exactly 1 placeholder
            # if you expect multiple occurrences, you'd need scatter logic.
            if pos.sum().item() == 0:
                continue

            # write [B,H] to the matched positions
            inputs_embeds[pos] = sensor_embeds[:, i, :].reshape(-1, H)

        return inputs_embeds

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, batch_x, padding_mask=None, *args, **kwargs):
        """
        batch_x: [B,L,C]
        padding_mask: [B,L] (optional, not used in prompt attention here)
        """
        B, L, C = batch_x.shape
        device = batch_x.device

        # 1) mask
        x_masked, mask = self._random_mask(batch_x)

        # 2) encode sensor -> [B,C,H]
        sensor_embeds = self._encode_sensor(x_masked)

        # 3) tokenize prompt -> [B,T]
        input_ids, attn_mask = self._tokenize_prompt(B, device)

        # 4) inject embeddings -> [B,T,H]
        inputs_embeds = self._inject_sensor_embeds(input_ids, sensor_embeds)

        # 5) run LLaMA
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_hidden_states=self.output_hidden_states,
            use_cache=False,
        )

        # out.logits: [B,T,V]
        # out.hidden_states: tuple of layers, each [B,T,H]
        return out, mask, x_masked
