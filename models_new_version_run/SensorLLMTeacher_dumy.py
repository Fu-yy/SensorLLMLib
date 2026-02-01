import torch
import os

from models_new_version_run.SensorLLMFuy_test_withllm_mae_vqvae import SoftLlamaTeacher, SoftLlamaTeacherLoRA

# === 配置 ===
LLAMA_PATH = "meta-llama/Llama-2-7b-hf"  # 你的基座模型路径
ADAPTER_PATH = "./checkpoints/lora_teacher_exp/final_adapter"  # 你的 LoRA 保存路径
NUM_CODES = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def check_teacher():
    # 伪造一个 codebook weight (只用于获取 shape)
    fake_codebook = torch.randn(NUM_CODES, 64)

    print(">>> 1. 初始化带 Adapter 的 Teacher...")
    teacher = SoftLlamaTeacherLoRA(
        llm_path=LLAMA_PATH,
        adapter_path=ADAPTER_PATH,
        codebook_weights=fake_codebook,
        mask_token_id=NUM_CODES,  # 512
        device=DEVICE
    )

    # 构造一个假输入： Batch=2, Length=10
    # 假设第 5 个位置是 MASK (512)
    dummy_seq = torch.randint(0, NUM_CODES, (2, 10)).to(DEVICE)
    dummy_seq[:, 5] = NUM_CODES  # 设置 Mask

    print("\n>>> 2. 运行推理...")
    out = teacher(dummy_seq)
    probs = out.probs  # [2, 512]

    # === 核心指标分析 ===
    print("\n====== Teacher Quality Report ======")

    # 1. Max Confidence (越接近 1 越好，说明不仅懂，而且确定)
    max_probs, preds = torch.max(probs, dim=-1)
    print(f"Max Probabilities (Top-1 Confidence): {max_probs.detach().cpu().numpy()}")
    print(f"Predicted Token Indices: {preds.detach().cpu().numpy()}")

    # 2. Entropy (越低越好，说明分布集中)
    # Entropy = - sum(p * log(p))
    # 均匀分布的 Entropy = log(512) ≈ 6.23
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    print(f"Entropy: {entropy.detach().cpu().numpy()}")

    avg_conf = max_probs.mean().item()
    avg_ent = entropy.mean().item()

    print("\n------------------------------------")
    if avg_conf > 0.1 and avg_ent < 5.0:
        print("✅ pass: Teacher 看起来很自信，LoRA 加载成功！")
        print("   (它学会了在 Sensor 空间内进行预测)")
    else:
        print("❌ fail: Teacher 输出接近均匀分布/噪声。")
        print("   可能原因：")
        print("   1. Adapter 路径不对或没加载上。")
        print("   2. LoRA 训练失败 (Loss 没降下来)。")
        print("   3. Prompt 格式和训练时不一致。")


if __name__ == "__main__":
    check_teacher()