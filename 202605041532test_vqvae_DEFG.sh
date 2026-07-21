#!/bin/bash
#set -e

# ===== 全局设置 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LLAMA_NAME="/root/autodl-tmp/Llama-3.2-1B"
TS_BACKBONE_YAML="ts_backbone.yaml"
TS_BACKBONE_YAML="ts_backbone_linux.yaml"


######################################################################
######################################################################
######################################################################
######################################################################
################## soft和temperature #################################
######################################################################
######################################################################
######################################################################







#!/bin/bash
#set -e

# ===== 全局设置 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LLAMA_NAME="/root/autodl-tmp/Llama-3.2-1B"
TS_BACKBONE_YAML="ts_backbone_linux.yaml"

MODEL_NAME="SensorLLMFuy_test_withllm_mae_vqvae"
VQVAEPATH="qua_recon_path"

MASK_RATE=0.4
MASK_MODE="random"

PRETRAIN_trainable_modules="student"
TRAIN_trainable_modules="student,pool_query,pool_attn,classifier"

# ===== 三组蒸馏实验 =====
# 格式：实验名|lambda_distill|use_hard_label|distill_temperature
EXPERIMENTS=(
  "A_soft_lambda0p1|0.1|0|2.0"
  "B_soft_lambda0p3|0.3|0|2.0"
  "C_soft_lambda0p5|0.5|0|2.0"
  "D_soft_lambda0p7|0.7|0|2.0"
  "E_soft_lambda0p9|0.9|0|2.0"
  "F_soft_lambda0p01|0.01|0|2.0"
  "G_soft_lambda0p03|0.03|0|2.0"
  "H_soft_lambda0p05|0.05|0|2.0"
  "I_soft_lambda0p07|0.07|0|2.0"
  "J_soft_lambda0p09|0.09|0|2.0"
  "K_hard_lambda0p3|0.3|1|1.0"
)
EXPERIMENTS=(
#"D_soft_lambda0p7|0.7|0|2.0"
#  "E_soft_lambda0p9|0.9|0|2.0"
#  "F_soft_lambda0p01|0.01|0|2.0"
  "G_soft_lambda0p03|0.03|0|2.0"
)
clean_cache() {
    TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v21"
    if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
        find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
        echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
    else
        echo "错误：目录不存在或变量为空，跳过删除"
    fi
}

run_stage1_stage2() {
    DATA_ROOT="$1"
    DATA_KEY="$2"
    DATA_NAME="$3"
    MODEL_ID="$4"
    RUN_ID="$5"
    LOG_DIR="$6"
    ALIGN_W_MAX="$7"
    SEQ_LEN="$8"
    PATCH_LEN="$9"
    BATCH_SIZE="${10}"
    LR="${11}"
    PRETRAIN_EPOCHS="${12}"
    EPOCHS="${13}"
    EXTRA_ARGS="${14}"

    mkdir -p "$LOG_DIR"

    echo "--------------------------------------------------------"
    echo "Running $DATA_NAME"
    echo "RUN_ID=$RUN_ID"
    echo "LOG_DIR=$LOG_DIR"
    echo "lambda_distill=$LAMBDA_DISTILL"
    echo "use_hard_label=$USE_HARD_LABEL"
    echo "distill_temperature=$DISTILL_TEMPERATURE"
    echo "--------------------------------------------------------"

    # Stage 1: teacher distillation / hard teacher / soft teacher
    python -u run.py \
      --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
      --model_id "$MODEL_ID" --run_id "$RUN_ID" --datasets "$DATA_NAME" \
      --model "$MODEL_NAME" --data "$DATA_NAME" --dataset_key "$DATA_KEY" \
      --seq_len "$ALIGN_W_MAX" --patch_len "$PATCH_LEN" --stride "$PATCH_LEN" --stage 1 \
      --batch_size "$BATCH_SIZE" --trainable_modules "$PRETRAIN_trainable_modules" \
      --llama_name "$LLAMA_NAME" \
      --learning_rate "$LR" --train_epochs "$PRETRAIN_EPOCHS" \
      --num_workers 0 --itr 1 \
      --use_hard_label "$USE_HARD_LABEL" \
      --mask_mode "$MASK_MODE" \
      --mask_rate "$MASK_RATE" \
      --lambda_distill "$LAMBDA_DISTILL" \
      --distill_temperature "$DISTILL_TEMPERATURE" \
      --ts_backbone_yaml "$TS_BACKBONE_YAML" \
      --vqvae_path "$VQVAEPATH" \
      $EXTRA_ARGS \
      > "$LOG_DIR/stage1.log" 2>&1

    # Stage 2: normal fine-tuning
    python -u run.py \
      --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
      --model_id "$MODEL_ID" --run_id "$RUN_ID" --datasets "$DATA_NAME" \
      --model "$MODEL_NAME" --data "$DATA_NAME" --dataset_key "$DATA_KEY" \
      --seq_len "$SEQ_LEN" --patch_len "$PATCH_LEN" --stride "$PATCH_LEN" --stage 2 \
      --batch_size "$BATCH_SIZE" --trainable_modules "$TRAIN_trainable_modules" \
      --llama_name "$LLAMA_NAME" \
      --learning_rate "$LR" --train_epochs "$EPOCHS" \
      --num_workers 0 --itr 5 \
      --use_hard_label "$USE_HARD_LABEL" \
      --mask_mode "$MASK_MODE" \
      --mask_rate "$MASK_RATE" \
      --lambda_distill "$LAMBDA_DISTILL" \
      --distill_temperature "$DISTILL_TEMPERATURE" \
      --ts_backbone_yaml "$TS_BACKBONE_YAML" \
      --vqvae_path "$VQVAEPATH" \
      $EXTRA_ARGS \
      > "$LOG_DIR/stage2.log" 2>&1

    echo "Done $DATA_NAME."
}

GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")

for EXP in "${EXPERIMENTS[@]}"; do
    IFS="|" read -r EXP_NAME LAMBDA_DISTILL USE_HARD_LABEL DISTILL_TEMPERATURE <<< "$EXP"

    GLOBAL_LOG_ROOT="./run_log/batch_${GLOBAL_TIME_TAG}_${EXP_NAME}_mask_${MASK_RATE}"
    mkdir -p "$GLOBAL_LOG_ROOT"

    echo "========================================================"
    echo "Starting experiment: $EXP_NAME"
    echo "lambda_distill=$LAMBDA_DISTILL"
    echo "use_hard_label=$USE_HARD_LABEL"
    echo "distill_temperature=$DISTILL_TEMPERATURE"
    echo "mask_mode=$MASK_MODE"
    echo "mask_rate=$MASK_RATE"
    echo "Logs: $GLOBAL_LOG_ROOT"
    echo "========================================================"

    # ========================================================
    # 1. UCIHAR
    # ========================================================
#    clean_cache
#    run_stage1_stage2 \
#      "/root/autodl-tmp/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset" \
#      "ucihar" \
#      "UCIHAR" \
#      "UCIHAR" \
#      "${GLOBAL_TIME_TAG}_ucihar_${EXP_NAME}" \
#      "$GLOBAL_LOG_ROOT/ucihar" \
#      128 \
#      128 \
#      64 \
#      32 \
#      0.001 \
#      20 \
#      8 \
#      ""

    # ========================================================
    # 2. USC-HAD
    # ========================================================
    clean_cache
    run_stage1_stage2 \
      "/root/autodl-tmp/datasets/USC-HAD/USC-HAD" \
      "uschad" \
      "USCHAD" \
      "USCHAD" \
      "${GLOBAL_TIME_TAG}_uschad_${EXP_NAME}" \
      "$GLOBAL_LOG_ROOT/uschad" \
      200 \
      200 \
      100 \
      16 \
      0.001 \
      20 \
      8 \
      "--test_subjects subject13,subject14"

    # ========================================================
    # 3. MHEALTH
    # ========================================================
    clean_cache
    run_stage1_stage2 \
      "/root/autodl-tmp/datasets/MHEALTHDATASET" \
      "mhealth" \
      "MHealth" \
      "MHealth" \
      "${GLOBAL_TIME_TAG}_mhealth_${EXP_NAME}" \
      "$GLOBAL_LOG_ROOT/mhealth" \
      100 \
      100 \
      50 \
      16 \
      0.001 \
      20 \
      8 \
      "--test_subjects subject1,subject3,subject6"

    # ========================================================
    # 4. PAMAP2 50Hz
    # ========================================================
    clean_cache
    run_stage1_stage2 \
      "/root/autodl-tmp/datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset" \
      "pamap50" \
      "PAMAP50" \
      "PAMAP2" \
      "${GLOBAL_TIME_TAG}_pamap50_${EXP_NAME}" \
      "$GLOBAL_LOG_ROOT/pamap50_50hz" \
      100 \
      100 \
      50 \
      32 \
      0.001 \
      20 \
      8 \
      "--pamap_variant pamap50 --test_subjects subject105,subject106"

    # ========================================================
    # 5. WISDM
    # ========================================================
    clean_cache
    run_stage1_stage2 \
      "/root/autodl-tmp/datasets/WISDM_ar_latest/WISDM_ar_v1.1" \
      "wisdm" \
      "WISDM" \
      "WISDM" \
      "${GLOBAL_TIME_TAG}_wisdm_${EXP_NAME}" \
      "$GLOBAL_LOG_ROOT/wisdm" \
      80 \
      80 \
      40 \
      64 \
      0.001 \
      20 \
      8 \
      "--test_users 33,34,35,36 --val_users 5,13,17,19,27,31 --wisdm_norm none"

    # ========================================================
    # 6. HHAR_1user
    # ========================================================
    clean_cache
    run_stage1_stage2 \
      "/root/autodl-tmp/datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp" \
      "hhar" \
      "HHAR_1user" \
      "HHAR_1user" \
      "${GLOBAL_TIME_TAG}_hhar_1user_${EXP_NAME}" \
      "$GLOBAL_LOG_ROOT/hhar_1user" \
      128 \
      128 \
      64 \
      64 \
      0.001 \
      20 \
      8 \
      "--hhar_tol 0.05 --hhar_align_on Arrival_Time --hhar_use_cache 1 --hhar_norm none"

    # ========================================================
    # 7. MotionSense
    # ========================================================
    clean_cache
    run_stage1_stage2 \
      "/root/autodl-tmp/datasets/motion-sense-master/motion-sense-master/data" \
      "motionsense" \
      "MotionSense" \
      "MotionSense" \
      "${GLOBAL_TIME_TAG}_motionsense_${EXP_NAME}" \
      "$GLOBAL_LOG_ROOT/motionsense" \
      128 \
      128 \
      64 \
      64 \
      0.001 \
      20 \
      8 \
      "--test_users 19,20,21,22,23,24 --val_users 13,14,15,16,17,18 --motionsense_feature_set A12 --motionsense_combine_grav_acc 0 --motionsense_norm none"

    echo "========================================================"
    echo "DONE experiment: $EXP_NAME"
    echo "Logs at: $GLOBAL_LOG_ROOT"
    echo "========================================================"
done




python /root/autodl-tmp/SensorLLMLib_v2/send_email.py


shutdown -h now
