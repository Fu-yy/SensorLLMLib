#!/bin/bash
# PrimitiveAlignHAR Stage1 -> Stage2 -> Test for each dataset
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

# profile 生成设置
AUTO_BUILD_PROFILE=1
FORCE_BUILD_PROFILE=0
PRIMITIVE_PROFILE_EXAMPLES=5
PRIMITIVE_PROFILE_MIN_VALID_RATIO=0.5

# 训练设置
BATCH_SIZE=32
NUM_WORKERS=0
TRAIN_EPOCHS_STAGE1=20
TRAIN_EPOCHS_STAGE2=20
LR_STAGE1="1e-4"
LR_STAGE2="1e-4"

# 模型结构
D_MODEL=128
N_HEADS=4
E_LAYERS=2
DROPOUT=0.1

# Stage1 多任务设置
LAMBDA_PRIMITIVE=0.5
ALIGN_MASK_RATE=0.3
SEMANTIC_WEIGHT=0.5
PRETRAIN_MONITOR="activity_acc"

# 是否 Stage2 训练完成后立即测试
RUN_TEST_AFTER_STAGE2=1

GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG_ROOT="./run_log/primalignhar_stage1_stage2_${GLOBAL_TIME_TAG}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting PrimitiveAlignHAR Stage1 -> Stage2 pipeline"
echo "Time tag: $GLOBAL_TIME_TAG"
echo "Logs: $GLOBAL_LOG_ROOT"
echo "LLM/Text Encoder: $LLAMA_NAME"
echo "VQ path key: $VQVAEPATH"
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
  RUN_ID="${GLOBAL_TIME_TAG}_${DATA_KEY}_primalignhar"

  LOG_DIR="$GLOBAL_LOG_ROOT/$LOG_SUBDIR"
  mkdir -p "$LOG_DIR"

  echo "========================================================"
  echo "Dataset Pipeline: $DATA_NAME"
  echo "DATA_KEY=$DATA_KEY"
  echo "RUN_ID=$RUN_ID"
  echo "SEQ_LEN=$SEQ_LEN PATCH_LEN=$PATCH_LEN"
  echo "LOG_DIR=$LOG_DIR"
  echo "========================================================"

  # ========================================================
  # Stage1: multi-task primitive alignment pretraining
  # ========================================================

  echo "--------------------------------------------------------"
  echo "[Stage1] Pretrain: $DATA_NAME"
  echo "--------------------------------------------------------"

  python -u run.py \
    --task_name alignment \
    --is_training 1 \
    --stage 1 \
    --root_path "$DATA_ROOT" \
    --model_id "$MODEL_ID" \
    --run_id "$RUN_ID" \
    --datasets "$DATA_NAME" \
    --model "$MODEL_NAME" \
    --data "$DATA_NAME" \
    --dataset_key "$DATA_KEY" \
    --seq_len "$SEQ_LEN" \
    --patch_len "$PATCH_LEN" \
    --stride "$PATCH_LEN" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --train_epochs "$TRAIN_EPOCHS_STAGE1" \
    --learning_rate "$LR_STAGE1" \
    --d_model "$D_MODEL" \
    --n_heads "$N_HEADS" \
    --e_layers "$E_LAYERS" \
    --dropout "$DROPOUT" \
    --lambda_primitive "$LAMBDA_PRIMITIVE" \
    --align_mask_rate "$ALIGN_MASK_RATE" \
    --semantic_weight "$SEMANTIC_WEIGHT" \
    --llama_name "$LLAMA_NAME" \
    --ts_backbone_yaml "$TS_BACKBONE_YAML" \
    --vqvae_path "$VQVAEPATH" \
    --auto_build_primitive_profile "$AUTO_BUILD_PROFILE" \
    --force_build_primitive_profile "$FORCE_BUILD_PROFILE" \
    --primitive_profile_examples "$PRIMITIVE_PROFILE_EXAMPLES" \
    --primitive_profile_min_valid_ratio "$PRIMITIVE_PROFILE_MIN_VALID_RATIO" \
    --pretrain_monitor "$PRETRAIN_MONITOR" \
    $EXTRA_ARGS \
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
  echo "[Stage2] Train: $DATA_NAME"
  echo "--------------------------------------------------------"

  python -u run.py \
    --task_name alignment \
    --is_training 1 \
    --stage 2 \
    --root_path "$DATA_ROOT" \
    --model_id "$MODEL_ID" \
    --run_id "$RUN_ID" \
    --datasets "$DATA_NAME" \
    --model "$MODEL_NAME" \
    --data "$DATA_NAME" \
    --dataset_key "$DATA_KEY" \
    --seq_len "$SEQ_LEN" \
    --patch_len "$PATCH_LEN" \
    --stride "$PATCH_LEN" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --train_epochs "$TRAIN_EPOCHS_STAGE2" \
    --learning_rate "$LR_STAGE2" \
    --d_model "$D_MODEL" \
    --n_heads "$N_HEADS" \
    --e_layers "$E_LAYERS" \
    --dropout "$DROPOUT" \
    --lambda_primitive "$LAMBDA_PRIMITIVE" \
    --align_mask_rate "$ALIGN_MASK_RATE" \
    --semantic_weight "$SEMANTIC_WEIGHT" \
    --llama_name "$LLAMA_NAME" \
    --ts_backbone_yaml "$TS_BACKBONE_YAML" \
    --vqvae_path "$VQVAEPATH" \
    --auto_build_primitive_profile "$AUTO_BUILD_PROFILE" \
    --force_build_primitive_profile 0 \
    --primitive_profile_examples "$PRIMITIVE_PROFILE_EXAMPLES" \
    --primitive_profile_min_valid_ratio "$PRIMITIVE_PROFILE_MIN_VALID_RATIO" \
    $EXTRA_ARGS \
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

    python -u run.py \
      --task_name alignment \
      --is_training 0 \
      --stage 2 \
      --root_path "$DATA_ROOT" \
      --model_id "$MODEL_ID" \
      --run_id "$RUN_ID" \
      --datasets "$DATA_NAME" \
      --model "$MODEL_NAME" \
      --data "$DATA_NAME" \
      --dataset_key "$DATA_KEY" \
      --seq_len "$SEQ_LEN" \
      --patch_len "$PATCH_LEN" \
      --stride "$PATCH_LEN" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --d_model "$D_MODEL" \
      --n_heads "$N_HEADS" \
      --e_layers "$E_LAYERS" \
      --dropout "$DROPOUT" \
      --lambda_primitive "$LAMBDA_PRIMITIVE" \
      --align_mask_rate "$ALIGN_MASK_RATE" \
      --semantic_weight "$SEMANTIC_WEIGHT" \
      --llama_name "$LLAMA_NAME" \
      --ts_backbone_yaml "$TS_BACKBONE_YAML" \
      --vqvae_path "$VQVAEPATH" \
      $EXTRA_ARGS \
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
MODEL_ID="UCIHAR_PrimitiveAlignHAR"
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
MODEL_ID="USCHAD_PrimitiveAlignHAR"
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
MODEL_ID="MHealth_PrimitiveAlignHAR"
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
MODEL_ID="PAMAP50_PrimitiveAlignHAR"
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
MODEL_ID="WISDM_PrimitiveAlignHAR"
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
MODEL_ID="HHAR_1user_PrimitiveAlignHAR"
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
MODEL_ID="MotionSense_PrimitiveAlignHAR"
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
echo "ALL PrimitiveAlignHAR Stage1 -> Stage2 pipelines DONE."
echo "Global time tag: $GLOBAL_TIME_TAG"
echo "Logs at: $GLOBAL_LOG_ROOT"
echo "========================================================"