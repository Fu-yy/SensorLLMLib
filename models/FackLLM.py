# models/fake_llm.py
import torch
import torch.nn as nn
from types import SimpleNamespace

class FakeLLM(nn.Module):
    def __init__(self, vocab_size=32000, hidden_size=256):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            pad_token_id=0,
            use_cache=False,
        )
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def get_input_embeddings(self):
        return self.emb

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        output_hidden_states=False,
        use_cache=False,
    ):
        if inputs_embeds is None:
            h = self.emb(input_ids)
        else:
            h = inputs_embeds

        h = self.proj(h)

        if output_hidden_states:
            return SimpleNamespace(hidden_states=[h])
        return SimpleNamespace(last_hidden_state=h)
