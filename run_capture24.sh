#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${TIME_TAG}_capture24_sensorllm"

LOG_ROOT="./run_log/log_${TIME_TAG}/capture24"
mkdir -p "$LOG_ROOT"

DATA_ROOT="./datasets/CAPTURE24/"
DATA_KEY="capture24"
DATA_NAME="CAPTURE24"

# ===== Paper settings =====
ALIGN_W_MIN=10
ALIGN_W_MAX=500
SEQ_LEN=500
PATCH_LEN=250

# paper: first 100 train, remaining 51 test, 5% sampled each participant
TRAIN_N=100
TEST_N=51
KEEP_RATIO=0.05
DOWNSAMPLE=2
LABEL_COL="label:WillettsSpecific2018"

BATCH_SIZE=8
LR=0.001
EPOCHS=8

model_name=SensorLLMFuy

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path "$DATA_ROOT" \
  --model_id CAPTURE24 \
  --run_id "$RUN_ID" \
  --datasets CAPTURE24Y \
  --model "$model_name" \
  --data $DATA_NAME \
  --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX \
  --patch_len $PATCH_LEN \
  --capture24_train_n $TRAIN_N \
  --capture24_test_n $TEST_N \
  --capture24_keep_ratio $KEEP_RATIO \
  --downsample_factor $DOWNSAMPLE \
  --capture24_label_col "$LABEL_COL" \
  --stage 1 \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --num_workers 0 \
  > "$LOG_ROOT/model_${model_name}_stage1.log" 2>&1

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path "$DATA_ROOT" \
  --model_id CAPTURE24 \
  --run_id "$RUN_ID" \
  --datasets CAPTURE24Y \
  --model "$model_name" \
  --data $DATA_NAME \
  --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN \
  --patch_len $PATCH_LEN \
  --capture24_train_n $TRAIN_N \
  --capture24_test_n $TEST_N \
  --capture24_keep_ratio $KEEP_RATIO \
  --downsample_factor $DOWNSAMPLE \
  --capture24_label_col "$LABEL_COL" \
  --stage 2 \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --num_workers 0 \
  > "$LOG_ROOT/model_${model_name}_stage2.log" 2>&1
