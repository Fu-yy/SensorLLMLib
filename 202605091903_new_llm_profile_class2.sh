#!/bin/bash
# PrimitiveAlignHAR Stage2 classification training
# 不建议 set -e，某个数据集失败时可以继续跑后面的
# set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ========================================================
# 全局设置
# ========================================================

#LLAMA_NAME="D:/fuy/MyCode/Qwen2.5-1.5B-Instruct"
 LLAMA_NAME="D:/fuy/MyCode/Llama-3.2-1B-Instruct"

TS_BACKBONE_YAML="ts_backbone.yaml"

VQVAEPATH="qua_recon_path"

MODEL_NAME="PrimitiveAlignHAR"

AUTO_BUILD_PROFILE=1
FORCE_BUILD_PROFILE=0
PRIMITIVE_PROFILE_EXAMPLES=5
PRIMITIVE_PROFILE_MIN_VALID_RATIO=0.5

BATCH_SIZE=32
TRAIN_EPOCHS=20
LR="1e-4"

D_MODEL=128
N_HEADS=4
E_LAYERS=2
DROPOUT=0.1
SEMANTIC_WEIGHT=0.5

GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG_ROOT="./run_log/primalignhar_stage2_train_${GLOBAL_TIME_TAG}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting PrimitiveAlignHAR Stage2 Train: $GLOBAL_TIME_TAG"
echo "Logs: $GLOBAL_LOG_ROOT"
echo "LLM/Text Encoder: $LLAMA_NAME"
echo "VQ path key: $VQVAEPATH"
echo "========================================================"


run_stage2_train () {
  DATA_ROOT="$1"
  DATA_KEY="$2"
  DATA_NAME="$3"
  MODEL_ID="$4"
  SEQ_LEN="$5"
  PATCH_LEN="$6"
  LOG_SUBDIR="$7"
  EXTRA_ARGS="$8"

  RUN_ID="${GLOBAL_TIME_TAG}_${DATA_KEY}_primalignhar_stage2"
  LOG_DIR="$GLOBAL_LOG_ROOT/$LOG_SUBDIR"
  mkdir -p "$LOG_DIR"

  echo "========================================================"
  echo "Stage2 Train: $DATA_NAME"
  echo "DATA_KEY=$DATA_KEY"
  echo "SEQ_LEN=$SEQ_LEN PATCH_LEN=$PATCH_LEN"
  echo "LOG_DIR=$LOG_DIR"
  echo "========================================================"

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
    --num_workers 0 \
    --train_epochs "$TRAIN_EPOCHS" \
    --learning_rate "$LR" \
    --d_model "$D_MODEL" \
    --n_heads "$N_HEADS" \
    --e_layers "$E_LAYERS" \
    --dropout "$DROPOUT" \
    --semantic_weight "$SEMANTIC_WEIGHT" \
    --llama_name "$LLAMA_NAME" \
    --ts_backbone_yaml "$TS_BACKBONE_YAML" \
    --vqvae_path "$VQVAEPATH" \
    --auto_build_primitive_profile "$AUTO_BUILD_PROFILE" \
    --force_build_primitive_profile "$FORCE_BUILD_PROFILE" \
    --primitive_profile_examples "$PRIMITIVE_PROFILE_EXAMPLES" \
    --primitive_profile_min_valid_ratio "$PRIMITIVE_PROFILE_MIN_VALID_RATIO" \
    $EXTRA_ARGS \
    > "$LOG_DIR/stage2_train.log" 2>&1

  echo "Done Stage2 Train: $DATA_NAME"
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

run_stage2_train \
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

run_stage2_train \
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

run_stage2_train \
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

run_stage2_train \
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

run_stage2_train \
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

run_stage2_train \
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

run_stage2_train \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "motionsense" \
  "--test_users 19,20,21,22,23,24 --val_users 13,14,15,16,17,18 --motionsense_feature_set A12 --motionsense_combine_grav_acc 0 --motionsense_norm none"


echo "========================================================"
echo "ALL PrimitiveAlignHAR Stage2 Train DONE."
echo "Logs at: $GLOBAL_LOG_ROOT"
echo "========================================================"