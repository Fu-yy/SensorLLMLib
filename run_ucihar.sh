#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${TIME_TAG}_ucihar_sensorllm"

LOG_ROOT="./run_log/log_${TIME_TAG}/ucihar"
mkdir -p "$LOG_ROOT"

DATA_ROOT="./datasets/UCIHAR/"
DATA_KEY="ucihar"
DATA_NAME="UCIHAR"

# ===== Paper settings =====
ALIGN_W_MIN=5
ALIGN_W_MAX=200
SEQ_LEN=128
PATCH_LEN=64

BATCH_SIZE=32
LR=0.001
EPOCHS=8

model_name=SensorLLMFuy

# Stage 1: seq_len=ALIGN_W_MAX (若暂不支持动态 w)
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path "$DATA_ROOT" \
  --model_id UCIHAR \
  --run_id "$RUN_ID" \
  --datasets UCIHARY \
  --model "$model_name" \
  --data $DATA_NAME \
  --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX \
  --patch_len $PATCH_LEN \
  --stage 1 \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --num_workers 0 \
  > "$LOG_ROOT/model_${model_name}_stage1.log" 2>&1

# Stage 2: HAR
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path "$DATA_ROOT" \
  --model_id UCIHAR \
  --run_id "$RUN_ID" \
  --datasets UCIHARY \
  --model "$model_name" \
  --data $DATA_NAME \
  --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN \
  --patch_len $PATCH_LEN \
  --stage 2 \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --num_workers 0 \
  > "$LOG_ROOT/model_${model_name}_stage2.log" 2>&1
