#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${TIME_TAG}_uschad_sensorllm"

LOG_ROOT="./run_log/log_${TIME_TAG}/uschad"
mkdir -p "$LOG_ROOT"

DATA_ROOT="./datasets/USC-HAD/"
DATA_KEY="uschad"
DATA_NAME="USCHAD"

# ===== Paper settings =====
ALIGN_W_MIN=5
ALIGN_W_MAX=200
SEQ_LEN=200          # HAR w
PATCH_LEN=100        # HAR stride

BATCH_SIZE=16
LR=0.001
EPOCHS=8
TEST_SUBJECTS="subject13 subject14"

model_name=SensorLLMFuy

# -------- Stage 1 (alignment) --------
# 若你代码支持动态窗口，把下面两行参数接到你的模型/loader里：
#   --align_w_min $ALIGN_W_MIN --align_w_max $ALIGN_W_MAX
# 如果暂时不支持，就先用 seq_len=ALIGN_W_MAX 跑通 alignment。
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path "$DATA_ROOT" \
  --model_id USCHAD \
  --run_id "$RUN_ID" \
  --datasets USCHADY \
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

# -------- Stage 2 (HAR) --------
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path "$DATA_ROOT" \
  --model_id USCHAD \
  --run_id "$RUN_ID" \
  --datasets USCHADY \
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
