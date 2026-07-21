#!/bin/bash
# Channel-Grounded PrimitiveAlignHAR Stage1 -> Stage2 -> Test for each dataset
# VQ primitive + online LLM + sample-adaptive prompt + channel/segment grounding
# 不建议 set -e，某个数据集失败时可以继续跑后面的
# set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ========================================================
# 全局设置
# ========================================================

LLAMA_NAME="D:/fuy/MyCode/Qwen2.5-1.5B-Instruct"
# LLAMA_NAME="D:/fuy/MyCode/Llama-3.2-1B-Instruct"

TS_BACKBONE_YAML="ts_backbone.yaml"

# 这里对应 yaml 里的 VQ-VAE 路径 key
# 可选：qua_path / freq_path / recon_path / qua_freq_path / qua_recon_path / freq_recon_path / all_path
VQVAEPATH="qua_recon_path"

MODEL_NAME="PrimitiveAlignHAR"

# ========================================================
# primitive profile 生成设置
# ========================================================

AUTO_BUILD_PROFILE=1
FORCE_BUILD_PROFILE=0
PRIMITIVE_PROFILE_EXAMPLES=5
PRIMITIVE_PROFILE_MIN_VALID_RATIO=0.5

# semantic cache 设置
# 第一次修改 primitive profile 或 LLM 后，可以设为 1
FORCE_BUILD_SEMANTIC_CACHE=0
SEMANTIC_BATCH_SIZE=16
SEMANTIC_MAX_LENGTH=128
PRIMITIVE_SEMANTIC_DIM=512

# ========================================================
# 训练设置
# ========================================================

# 在线 LLM 比小 Transformer 慢很多。
# 如果显存不够，先改成 8 或 4。
BATCH_SIZE=16
NUM_WORKERS=0

TRAIN_EPOCHS_STAGE1=20
TRAIN_EPOCHS_STAGE2=8

LR_STAGE1="1e-4"
LR_STAGE2="1e-4"

# 是否 Stage2 训练完成后立即测试
RUN_TEST_AFTER_STAGE2=0

# ========================================================
# 模型结构
# ========================================================

D_MODEL=128
N_HEADS=4
E_LAYERS=2
DROPOUT=0.1

# ========================================================
# Stage1 多任务设置
# ========================================================

# 原始 masked primitive id recovery
LAMBDA_PRIMITIVE=0.5
ALIGN_MASK_RATE=0.3

# primitive semantic profile 权重
SEMANTIC_WEIGHT=0.5

# 新增：masked primitive semantic reconstruction
LAMBDA_SEMANTIC=0.2

# 新增：label-text contrastive alignment
LAMBDA_LABEL_ALIGN=0.2
LABEL_ALIGN_TEMPERATURE=0.07

# Stage1 early stopping / checkpoint monitor
# 可选：activity_acc / primitive_acc / label_align_acc / loss
PRETRAIN_MONITOR="activity_acc"

# ========================================================
# Channel-grounded primitive augmentation
# ========================================================

# 每个 primitive token 加当前样本局部统计：
# per-channel mean/std/energy + acc_mag mean/std + gyro_mag mean/std
USE_SEGMENT_STATS=1
STAT_WEIGHT=1.0

# 在 primitive tokens 前加入每个通道的 summary token：
# channel mean/std/min/max/energy/abs_mean/slope
USE_CHANNEL_SUMMARY=1

# ========================================================
# Sample-adaptive prompt 设置
# ========================================================

# 1: 使用文本 prompt
# 0: 完全不用文本 prompt，只用 embedding tokens
USE_TEXT_PROMPT=1

# 1: 每个样本动态生成 prompt，把 mean/std/energy/slope/low_motion/primitive diversity 写入文本
# 0: 使用固定 task-level prompt
USE_SAMPLE_PROMPT=1

# 动态 prompt 一般比固定 prompt 长，建议 256/384/512
PROMPT_MAX_LENGTH=384

# 动态 prompt 中展示能量最高的前几个通道
PROMPT_TOP_CHANNELS=3

# 动态 prompt 中展示几个 primitive segment 的局部统计
PROMPT_NUM_SEGMENT_EXAMPLES=4

# 是否把 primitive id 序列的一部分直接写进 prompt
# 0 更稳，避免模型过度依赖 id 文本；1 可做消融
PROMPT_INCLUDE_PRIMITIVE_IDS=0

# label text embedding 构建设置
LABEL_TEXT_BATCH_SIZE=16
LABEL_TEXT_MAX_LENGTH=64



use_static_branch=1
static_alpha=0.8
lambda_static_aux=0.2
lambda_static_gate=0.05

# ========================================================
# Stage2 微调设置
# ========================================================

# 可选：
# activity_only      只训分类头
# activity_projector 训投影层 + 分类头
# adapter_all        训所有 adapter/head，不训 VQ-VAE 和 LLM
STAGE2_TRAINABLE="activity_projector"

# ========================================================
# 日志设置
# ========================================================

GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG_ROOT="./run_log/primalignhar_cg_sample_prompt_${GLOBAL_TIME_TAG}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting Channel-Grounded PrimitiveAlignHAR pipeline"
echo "Time tag: $GLOBAL_TIME_TAG"
echo "Logs: $GLOBAL_LOG_ROOT"
echo "LLM: $LLAMA_NAME"
echo "VQ path key: $VQVAEPATH"
echo "Model: $MODEL_NAME"
echo "Use sample prompt: $USE_SAMPLE_PROMPT"
echo "Use segment stats: $USE_SEGMENT_STATS"
echo "Use channel summary: $USE_CHANNEL_SUMMARY"
echo "========================================================"


# ========================================================
# 通用函数：一个数据集先 Stage1，再 Stage2，再 Test
# ========================================================

run_dataset_pipeline () {
  DATA_ROOT="$1"
  DATA_KEY="$2"
  DATA_NAME="$3"
  MODEL_ID="$4"
  SEQ_LEN="$5"
  PATCH_LEN="$6"
  LOG_SUBDIR="$7"
  EXTRA_ARGS="$8"

  # 关键：Stage1 和 Stage2 必须使用同一个 RUN_ID
  RUN_ID="${GLOBAL_TIME_TAG}_${DATA_KEY}_cgprimalignhar"

  LOG_DIR="$GLOBAL_LOG_ROOT/$LOG_SUBDIR"
  mkdir -p "$LOG_DIR"

  echo "========================================================"
  echo "Dataset Pipeline: $DATA_NAME"
  echo "DATA_KEY=$DATA_KEY"
  echo "RUN_ID=$RUN_ID"
  echo "SEQ_LEN=$SEQ_LEN PATCH_LEN=$PATCH_LEN"
  echo "LOG_DIR=$LOG_DIR"
  echo "========================================================"

  COMMON_ARGS="\
    --task_name alignment \
    --root_path \"$DATA_ROOT\" \
    --model_id \"$MODEL_ID\" \
    --run_id \"$RUN_ID\" \
    --datasets \"$DATA_NAME\" \
    --model \"$MODEL_NAME\" \
    --data \"$DATA_NAME\" \
    --dataset_key \"$DATA_KEY\" \
    --seq_len \"$SEQ_LEN\" \
    --patch_len \"$PATCH_LEN\" \
    --stride \"$PATCH_LEN\" \
    --batch_size \"$BATCH_SIZE\" \
    --num_workers \"$NUM_WORKERS\" \
    --d_model \"$D_MODEL\" \
    --n_heads \"$N_HEADS\" \
    --e_layers \"$E_LAYERS\" \
    --dropout \"$DROPOUT\" \
    --llama_name \"$LLAMA_NAME\" \
    --ts_backbone_yaml \"$TS_BACKBONE_YAML\" \
    --vqvae_path \"$VQVAEPATH\" \
    --primitive_semantic_dim \"$PRIMITIVE_SEMANTIC_DIM\" \
    --semantic_batch_size \"$SEMANTIC_BATCH_SIZE\" \
    --semantic_max_length \"$SEMANTIC_MAX_LENGTH\" \
    --force_build_semantic_cache \"$FORCE_BUILD_SEMANTIC_CACHE\" \
    --lambda_primitive \"$LAMBDA_PRIMITIVE\" \
    --align_mask_rate \"$ALIGN_MASK_RATE\" \
    --semantic_weight \"$SEMANTIC_WEIGHT\" \
    --lambda_semantic \"$LAMBDA_SEMANTIC\" \
    --lambda_label_align \"$LAMBDA_LABEL_ALIGN\" \
    --label_align_temperature \"$LABEL_ALIGN_TEMPERATURE\" \
    --label_text_batch_size \"$LABEL_TEXT_BATCH_SIZE\" \
    --label_text_max_length \"$LABEL_TEXT_MAX_LENGTH\" \
    --use_segment_stats \"$USE_SEGMENT_STATS\" \
    --stat_weight \"$STAT_WEIGHT\" \
    --use_channel_summary \"$USE_CHANNEL_SUMMARY\" \
    --use_text_prompt \"$USE_TEXT_PROMPT\" \
    --use_sample_prompt \"$USE_SAMPLE_PROMPT\" \
    --prompt_max_length \"$PROMPT_MAX_LENGTH\" \
    --prompt_top_channels \"$PROMPT_TOP_CHANNELS\" \
    --prompt_num_segment_examples \"$PROMPT_NUM_SEGMENT_EXAMPLES\" \
    --prompt_include_primitive_ids \"$PROMPT_INCLUDE_PRIMITIVE_IDS\" \
    $EXTRA_ARGS"

  # ========================================================
  # Stage1: channel-grounded primitive-language alignment pretraining
  # ========================================================

  echo "--------------------------------------------------------"
  echo "[Stage1] Channel-grounded primitive-language pretrain: $DATA_NAME"
  echo "--------------------------------------------------------"

  eval python -u run.py \
    $COMMON_ARGS \
    --is_training 1 \
    --stage 1 \
    --train_epochs "$TRAIN_EPOCHS_STAGE1" \
    --learning_rate "$LR_STAGE1" \
    --itr 1 \
    --use_static_branch $use_static_branch \
    --static_alpha $static_alpha \
    --lambda_static_aux $lambda_static_aux \
    --lambda_static_gate $lambda_static_gate \
    --auto_build_primitive_profile "$AUTO_BUILD_PROFILE" \
    --force_build_primitive_profile "$FORCE_BUILD_PROFILE" \
    --primitive_profile_examples "$PRIMITIVE_PROFILE_EXAMPLES" \
    --primitive_profile_min_valid_ratio "$PRIMITIVE_PROFILE_MIN_VALID_RATIO" \
    --pretrain_monitor "$PRETRAIN_MONITOR" \
    > "$LOG_DIR/stage1_pretrain.log" 2>&1

  STAGE1_STATUS=$?

  if [ $STAGE1_STATUS -ne 0 ]; then
    echo "[ERROR] Stage1 failed for $DATA_NAME. See: $LOG_DIR/stage1_pretrain.log"
    return
  fi

  echo "[OK] Stage1 done: $DATA_NAME"


  # ========================================================
  # Stage2: classification fine-tuning
  # ========================================================

  echo "--------------------------------------------------------"
  echo "[Stage2] Classification fine-tuning: $DATA_NAME"
  echo "--------------------------------------------------------"

  eval python -u run.py \
    $COMMON_ARGS \
    --is_training 1 \
    --stage 2 \
    --train_epochs "$TRAIN_EPOCHS_STAGE2" \
    --learning_rate "$LR_STAGE2" \
    --itr 5 \
    --learning_rate 0.00001 \
    --use_static_branch $use_static_branch \
    --static_alpha $static_alpha \
    --lambda_static_aux $lambda_static_aux \
    --lambda_static_gate $lambda_static_gate \
    --stage2_trainable "$STAGE2_TRAINABLE" \
    --auto_build_primitive_profile "$AUTO_BUILD_PROFILE" \
    --force_build_primitive_profile 0 \
    --primitive_profile_examples "$PRIMITIVE_PROFILE_EXAMPLES" \
    --primitive_profile_min_valid_ratio "$PRIMITIVE_PROFILE_MIN_VALID_RATIO" \
    > "$LOG_DIR/stage2_train.log" 2>&1

  STAGE2_STATUS=$?

  if [ $STAGE2_STATUS -ne 0 ]; then
    echo "[ERROR] Stage2 train failed for $DATA_NAME. See: $LOG_DIR/stage2_train.log"
    return
  fi

  echo "[OK] Stage2 train done: $DATA_NAME"


  # ========================================================
  # Stage2 Test
  # ========================================================

  if [ "$RUN_TEST_AFTER_STAGE2" -eq 1 ]; then
    echo "--------------------------------------------------------"
    echo "[Stage2] Test: $DATA_NAME"
    echo "--------------------------------------------------------"

    eval python -u run.py \
      $COMMON_ARGS \
      --is_training 0 \
      --stage 2 \
      --stage2_trainable "$STAGE2_TRAINABLE" \
      --auto_build_primitive_profile "$AUTO_BUILD_PROFILE" \
      --force_build_primitive_profile 0 \
      > "$LOG_DIR/stage2_test.log" 2>&1

    TEST_STATUS=$?

    if [ $TEST_STATUS -ne 0 ]; then
      echo "[ERROR] Stage2 test failed for $DATA_NAME. See: $LOG_DIR/stage2_test.log"
      return
    fi

    echo "[OK] Stage2 test done: $DATA_NAME"
  fi

  echo "========================================================"
  echo "[DONE] Dataset finished: $DATA_NAME"
  echo "RUN_ID=$RUN_ID"
  echo "Logs: $LOG_DIR"
  echo "========================================================"
}


# ========================================================
# 1. UCIHAR
# ========================================================

DATA_ROOT="D:/fuy/MyCode/SensorLLMLib/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
DATA_KEY="ucihar"
DATA_NAME="UCIHAR"
MODEL_ID="UCIHAR_CGPrimitiveAlignHAR"
SEQ_LEN=128
PATCH_LEN=64

run_dataset_pipeline \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "ucihar" \
  ""


# ========================================================
# 2. USC-HAD
# ========================================================

DATA_ROOT="D:/fuy/MyCode/SensorLLMLib/datasets/USC-HAD/USC-HAD"
DATA_KEY="uschad"
DATA_NAME="USCHAD"
MODEL_ID="USCHAD_CGPrimitiveAlignHAR"
SEQ_LEN=200
PATCH_LEN=100
TEST_SUBJECTS="subject13,subject14"

run_dataset_pipeline \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "uschad" \
  "--test_subjects $TEST_SUBJECTS"


# ========================================================
# 3. MHEALTH
# ========================================================

DATA_ROOT="D:/fuy/MyCode/SensorLLMLib/datasets/MHEALTHDATASET"
DATA_KEY="mhealth"
DATA_NAME="MHealth"
MODEL_ID="MHealth_CGPrimitiveAlignHAR"
SEQ_LEN=100
PATCH_LEN=50
TEST_SUBJECTS="subject1,subject3,subject6"

run_dataset_pipeline \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "mhealth" \
  "--test_subjects $TEST_SUBJECTS"


# ========================================================
# 4. PAMAP2 50Hz
# ========================================================

DATA_ROOT="D:/fuy/MyCode/SensorLLMLib/datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"
DATA_KEY="pamap50"
DATA_NAME="PAMAP50"
MODEL_ID="PAMAP50_CGPrimitiveAlignHAR"
SEQ_LEN=100
PATCH_LEN=50
PAMAP_VARIANT="pamap50"
TEST_SUBJECTS="subject105,subject106"

run_dataset_pipeline \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "pamap50" \
  "--pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS"


# ========================================================
# 5. WISDM
# ========================================================

DATA_ROOT="D:/fuy/MyCode/SensorLLMLib/datasets/WISDM_ar_latest/WISDM_ar_v1.1"
DATA_KEY="wisdm"
DATA_NAME="WISDM"
MODEL_ID="WISDM_CGPrimitiveAlignHAR"
SEQ_LEN=80
PATCH_LEN=40

run_dataset_pipeline \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "wisdm" \
  "--test_users 33,34,35,36 --val_users 5,13,17,19,27,31 --wisdm_norm none"


# ========================================================
# 6. HHAR_1user
# ========================================================

DATA_ROOT="D:/fuy/MyCode/SensorLLMLib/datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"
DATA_KEY="hhar"
DATA_NAME="HHAR_1user"
MODEL_ID="HHAR_1user_CGPrimitiveAlignHAR"
SEQ_LEN=128
PATCH_LEN=64

run_dataset_pipeline \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "hhar_1user" \
  "--hhar_tol 0.05 --hhar_align_on Arrival_Time --hhar_use_cache 1 --hhar_norm none"


# ========================================================
# 7. MotionSense
# ========================================================

DATA_ROOT="D:/fuy/MyCode/SensorLLMLib/datasets/motion-sense-master/motion-sense-master/data"
DATA_KEY="motionsense"
DATA_NAME="MotionSense"
MODEL_ID="MotionSense_CGPrimitiveAlignHAR"
SEQ_LEN=128
PATCH_LEN=64

run_dataset_pipeline \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "motionsense" \
  "--test_users 19,20,21,22,23,24 --val_users 13,14,15,16,17,18 --motionsense_feature_set A12 --motionsense_combine_grav_acc 0 --motionsense_norm none"


echo "========================================================"
echo "ALL Channel-Grounded PrimitiveAlignHAR pipelines DONE."
echo "Global time tag: $GLOBAL_TIME_TAG"
echo "Logs at: $GLOBAL_LOG_ROOT"
echo "========================================================"