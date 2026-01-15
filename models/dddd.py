import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class SensorPatchTokenizer(nn.Module):
    """
    1) token化：把 [B, L, C] 切成 patches -> [B, N, P*C]
    """
    def __init__(self, patch_len: int):
        super().__init__()
        self.patch_len = patch_len

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        P = self.patch_len
        pad = (P - (L % P)) % P
        if pad > 0:
            x = torch.cat([torch.zeros(B, pad, C, device=x.device, dtype=x.dtype), x], dim=1)
            L = L + pad

        N = L // P
        x = x.view(B, N, P, C).contiguous()     # [B, N, P, C]
        x = x.view(B, N, P * C).contiguous()    # [B, N, P*C]
        return x  # tokens


class Sensor2LlamaAdapter(nn.Module):
    """
    2) 你说的 linear：把 patch tokens -> Llama hidden size
       等价于：先做 sensor embedding，再映射到 Llama embedding 空间
    """
    def __init__(self, in_dim: int, llama_hidden: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, llama_hidden)

    def forward(self, sensor_tokens):
        # sensor_tokens: [B, N, in_dim]
        return self.proj(sensor_tokens)  # [B, N, llama_hidden]


class LlamaForHAR(nn.Module):
    """
    3) Llama attention 参与推理
    4) head 输出活动类别
    """
    def __init__(self, llama_name: str, n_classes: int, patch_len: int, sensor_channels: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llama_name, use_fast=True)
        self.llm = AutoModelForCausalLM.from_pretrained(llama_name)

        llama_hidden = self.llm.config.hidden_size

        # 1) tokenize
        self.ts_tokenizer = SensorPatchTokenizer(patch_len=patch_len)

        # 2) linear adapter：这里 in_dim = patch_len * C（最简单版本）
        in_dim = patch_len * sensor_channels
        self.adapter = Sensor2LlamaAdapter(in_dim=in_dim, llama_hidden=llama_hidden)

        # 4) 分类头：用最后一个 token 或 mean pooling
        self.cls_head = nn.Linear(llama_hidden, n_classes)




        # 可选：冻结 Llama，只训练 adapter + cls_head（推荐先这样跑通）
        for p in self.llm.parameters():
            p.requires_grad = False

    def forward(self, x_ts, prompt_text=None):
        """
        x_ts: [B, L, C]
        prompt_text: 可选文本提示（比如 "Classify the activity:"）
        """
        B = x_ts.size(0)

        # ---- 1) tokenize ----
        sensor_tokens = self.ts_tokenizer(x_ts)          # [B, N, P*C]

        # ---- 1.5) 你要的魔改：插 Linear ----
        sensor_soft_embeds = self.adapter(sensor_tokens) # [B, N, H]

        # ---- 文本 prompt embeds（可选）----
        if prompt_text is None:
            prompt_text = "Classify the activity:"
        tok = self.tokenizer([prompt_text] * B, return_tensors="pt", padding=True)
        tok = {k: v.to(x_ts.device) for k, v in tok.items()}

        text_embeds = self.llm.model.embed_tokens(tok["input_ids"])  # [B, T, H]

        # ---- 拼接： [sensor_soft_tokens] + [text_tokens] ----
        inputs_embeds = torch.cat([sensor_soft_embeds, text_embeds], dim=1)  # [B, N+T, H]

        # attention mask 也要拼：sensor 部分全 1
        attn_sensor = torch.ones(B, sensor_soft_embeds.size(1), device=x_ts.device, dtype=tok["attention_mask"].dtype)
        attention_mask = torch.cat([attn_sensor, tok["attention_mask"]], dim=1)  # [B, N+T]

        # ---- 3) 走 Llama attention ----
        out = self.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )

        last_hidden = out.last_hidden_state  # [B, N+T, H]

        # ---- 4) 分类：两种常用做法二选一 ----
        # (a) 用最后一个位置：
        feat = last_hidden[:, -1, :]  # [B, H]
        # (b) 或对 sensor token 做 mean pooling（更像 HAR）：
        # feat = last_hidden[:, :sensor_soft_embeds.size(1), :].mean(dim=1)

        logits = self.cls_head(feat)  # [B, n_classes]
        return logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO 你来填：mHealth 的通道数C、类别数、patch_len
    C = 24          # 例子：mHealth 可能不是 24，你要换成你真实的
    n_classes = 12  # 例子
    patch_len = 16  # 例子

    model = LlamaForHAR(
        llama_name="meta-llama/Llama-3.2-1B",  # TODO 换成你本地/实际模型名
        n_classes=n_classes,
        patch_len=patch_len,
        sensor_channels=C,
    ).to(device)

    # 假输入
    x = torch.randn(8, 256, C).to(device)  # [B, L, C]
    logits = model(x)
    print(logits.shape)  # [8, n_classes]
