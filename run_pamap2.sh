#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${TIME_TAG}_pamap2_sensorllm"

LOG_ROOT="./run_log/log_${TIME_TAG}/pamap2"
mkdir -p "$LOG_ROOT"

DATA_ROOT="./datasets/PAMAP2/"
DATA_KEY="pamap2"
DATA_NAME="PAMAP2"

# ===== Paper settings =====
ALIGN_W_MIN=5
ALIGN_W_MAX=100
SEQ_LEN=100
PATCH_LEN=50

# paper: downsample 100->50Hz
PAMAP_VARIANT="pamap50"
TEST_SUBJECTS="subject105 subject106"

BATCH_SIZE=32
LR=0.001
EPOCHS=8

model_name=SensorLLMFuy

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path "$DATA_ROOT" \
  --model_id PAMAP2 \
  --run_id "$RUN_ID" \
  --datasets PAMAP2Y \
  --model "$model_name" \
  --data $DATA_NAME \
  --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT \
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
  --model_id PAMAP2 \
  --run_id "$RUN_ID" \
  --datasets PAMAP2Y \
  --model "$model_name" \
  --data $DATA_NAME \
  --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN \
  --patch_len $PATCH_LEN \
  --stage 2 \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --num_workers 0 \
  > "$LOG_ROOT/model_${model_name}_stage2.log" 2>&1



















#---------------------------- PAMAP _____________________________

#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${TIME_TAG}_pamap2_100hz_sensorllm"

LOG_ROOT="./run_log/log_${TIME_TAG}/pamap2_100hz"
mkdir -p "$LOG_ROOT"

DATA_ROOT="./datasets/PAMAP2/"
DATA_KEY="pamap2"
DATA_NAME="PAMAP2"

# =========================
# Paper settings (PAMAP2 100Hz)
# =========================
ALIGN_W_MIN=5
ALIGN_W_MAX=100        # alignment stage uses variable w ∈ [5,100]

SEQ_LEN=200            # HAR window w = 200
PATCH_LEN=100          # stride = 100

PAMAP_VARIANT="pamap"  # <<< 原始 100Hz
TEST_SUBJECTS="subject105 subject106"

BATCH_SIZE=32
LR=0.001
EPOCHS=8

model_name=SensorLLMFuy

# =========================
# Stage 1: Alignment / Pretrain
# =========================
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path "$DATA_ROOT" \
  --model_id PAMAP2 \
  --run_id "$RUN_ID" \
  --datasets PAMAP2Y \
  --model "$model_name" \
  --data $DATA_NAME \
  --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX \
  --patch_len $PATCH_LEN \
  --stage 1 \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --num_workers 0 \
  > "$LOG_ROOT/model_${model_name}_stage1.log" 2>&1

# =========================
# Stage 2: HAR Finetune
# =========================
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path "$DATA_ROOT" \
  --model_id PAMAP2 \
  --run_id "$RUN_ID" \
  --datasets PAMAP2Y \
  --model "$model_name" \
  --data $DATA_NAME \
  --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN \
  --patch_len $PATCH_LEN \
  --stage 2 \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --num_workers 0 \
  > "$LOG_ROOT/model_${model_name}_stage2.log" 2>&1
