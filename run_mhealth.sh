#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${TIME_TAG}_mhealth_sensorllm"

LOG_ROOT="./run_log/log_${TIME_TAG}/mhealth"
mkdir -p "$LOG_ROOT"

DATA_ROOT="./datasets/MHEALTHDATASET/"
DATA_KEY="mhealth"
DATA_NAME="MHealth"

# ===== Paper settings =====
ALIGN_W_MIN=5
ALIGN_W_MAX=100
SEQ_LEN=100
PATCH_LEN=50

TEST_SUBJECTS="subject1 subject3 subject6"

BATCH_SIZE=16
LR=0.001
EPOCHS=8

model_name=SensorLLMFuy

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path "$DATA_ROOT" \
  --model_id MHealth \
  --run_id "$RUN_ID" \
  --datasets MHEALTHY \
  --model "$model_name" \
  --data $DATA_NAME \
  --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX \
  --patch_len $PATCH_LEN \
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
  --model_id MHealth \
  --run_id "$RUN_ID" \
  --datasets MHEALTHY \
  --model "$model_name" \
  --data $DATA_NAME \
  --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN \
  --patch_len $PATCH_LEN \
  --stage 2 \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --num_workers 0 \
  > "$LOG_ROOT/model_${model_name}_stage2.log" 2>&1
