#!/bin/bash
#set -e  # 遇到错误立即停止。如果希望忽略错误继续跑下一个，请注释掉这一行

# ===== 全局设置 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 生成一个全局时间标签，这样这一次批量运行的所有日志都在同一个大目录下
GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG_ROOT="./run_log/batch_${GLOBAL_TIME_TAG}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting Batch Training Run: $GLOBAL_TIME_TAG"
echo "Logs will be saved to: $GLOBAL_LOG_ROOT"
echo "========================================================"


LLAMA_NAME="D:\fuy\MyCode/Llama-3.2-1B"
model_name=SensorLLMFuy
PRETRAIN_trainable_modules="sensor_patch_proj,channel_id,patch_pos,mask_embed,recon_head,recon_scale_log"
TRAIN_trainable_modules="sensor_patch_proj,channel_id,patch_pos,pool_query,pool_attn,cls_head"


# ========================================================
# 1. UCIHAR
# ========================================================
echo "[1/5] Running UCIHAR..."
DATA_ROOT="D:\fuy\MyCode/SensorLLMLib/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
DATA_KEY="ucihar"
DATA_NAME="UCIHAR"
RUN_ID="${GLOBAL_TIME_TAG}_ucihar_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/ucihar"
mkdir -p "$LOG_DIR"

# Settings
ALIGN_W_MAX=200
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=32
LR=0.001
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done UCIHAR."

# ========================================================
# 2. USC-HAD
# ========================================================
echo "[2/5] Running USC-HAD..."
DATA_ROOT="D:\fuy\MyCode/SensorLLMLib/datasets/USC-HAD/USC-HAD"
DATA_KEY="uschad"
DATA_NAME="USCHAD"
RUN_ID="${GLOBAL_TIME_TAG}_uschad_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/uschad"
mkdir -p "$LOG_DIR"

# Settings (重置变量)
ALIGN_W_MAX=200
SEQ_LEN=200
PATCH_LEN=100
BATCH_SIZE=16 # 注意这里变了
LR=0.001
EPOCHS=8
TEST_SUBJECTS="subject13,subject14"

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id USCHAD --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id USCHAD --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done USC-HAD."

# ========================================================
# 3. MHEALTH
# ========================================================
echo "[3/5] Running MHEALTH..."
DATA_ROOT="D:\fuy\MyCode/SensorLLMLib/datasets/MHEALTHDATASET"
DATA_KEY="mhealth"
DATA_NAME="MHealth"
RUN_ID="${GLOBAL_TIME_TAG}_mhealth_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/mhealth"
mkdir -p "$LOG_DIR"

# Settings
ALIGN_W_MAX=100
SEQ_LEN=100
PATCH_LEN=50
TEST_SUBJECTS="subject1,subject3,subject6"
BATCH_SIZE=16

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MHealth --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MHealth --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done MHEALTH."

# ========================================================
# 4. PAMAP2 (50Hz Variant)
# ========================================================
echo "[4/5] Running PAMAP2 (50Hz)..."
DATA_ROOT="D:\fuy\MyCode/SensorLLMLib/datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"
DATA_KEY="pamap50"
DATA_NAME="PAMAP50"
RUN_ID="${GLOBAL_TIME_TAG}_pamap50_50hz_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/pamap50_50hz"
mkdir -p "$LOG_DIR"

# Settings
ALIGN_W_MAX=100
SEQ_LEN=100
PATCH_LEN=50
PAMAP_VARIANT="pamap50"
TEST_SUBJECTS="subject105,subject106"
BATCH_SIZE=32

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id PAMAP2 --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id PAMAP2 --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done PAMAP50 (50Hz)."

# ========================================================
# 5. PAMAP2 (100Hz Variant)
# ========================================================
echo "[5/5] Running PAMAP (100Hz)..."
# 注意：DATA_ROOT, KEY, NAME 复用上面的，但 LOG_DIR 和 RUN_ID 变了

DATA_ROOT="D:\fuy\MyCode/SensorLLMLib/datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"
DATA_KEY="pamap"
DATA_NAME="PAMAP"
RUN_ID="${GLOBAL_TIME_TAG}_pamap_100hz_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/pamap_100hz"
mkdir -p "$LOG_DIR"

# Settings
ALIGN_W_MAX=100
SEQ_LEN=200
PATCH_LEN=100
PAMAP_VARIANT="pamap" # 原始 100Hz
TEST_SUBJECTS="subject105,subject106"
BATCH_SIZE=32



# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id PAMAP2 --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id PAMAP2 --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done PAMAP (100Hz)."

# ========================================================
# Optional: CAPTURE24 (Skipped)
# ========================================================
if false; then
# ========================================================
# 6. CAPTURE24
# ========================================================
echo "[6/6] Running CAPTURE24..."

# 1. 继承全局路径和时间设置
# 注意：这里我们使用 GLOBAL_TIME_TAG 确保和其他数据集在同一个大文件夹下
RUN_ID="${GLOBAL_TIME_TAG}_capture24_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/capture24"
mkdir -p "$LOG_DIR"

DATA_ROOT="D:\fuy\MyCode/SensorLLMLib/datasets/capture24/capture24"
DATA_KEY="capture24"
DATA_NAME="CAPTURE24"

# 2. Paper Settings
ALIGN_W_MAX=500
SEQ_LEN=500
PATCH_LEN=250

# Capture24 specific
TRAIN_N=100
TEST_N=51
KEEP_RATIO=0.05
DOWNSAMPLE=2
LABEL_COL="label:WillettsSpecific2018"

BATCH_SIZE=8
LR=0.001
EPOCHS=8
model_name=SensorLLMFuy

# 3. Stage 1 (Alignment)
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path "$DATA_ROOT" \
  --model_id CAPTURE24 \
  --run_id "$RUN_ID" \
  --datasets $DATA_NAME \
  --model "$model_name" \
  --data $DATA_NAME \
  --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX \
  --patch_len $PATCH_LEN --stride $PATCH_LEN \
  --capture24_train_n $TRAIN_N \
  --capture24_test_n $TEST_N \
  --capture24_keep_ratio $KEEP_RATIO \
  --downsample_factor $DOWNSAMPLE \
  --capture24_label_col "$LABEL_COL" \
  --stage 1 \
  --trainable_modules $PRETRAIN_trainable_modules \
  --batch_size $BATCH_SIZE \
  --llama_name $LLAMA_NAME \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --num_workers 10 \
  > "$LOG_DIR/stage1.log" 2>&1

# 4. Stage 2 (HAR)
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path "$DATA_ROOT" \
  --model_id CAPTURE24 \
  --run_id "$RUN_ID" \
  --datasets $DATA_NAME \
  --model "$model_name" \
  --data $DATA_NAME \
  --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN \
  --patch_len $PATCH_LEN --stride $PATCH_LEN \
  --capture24_train_n $TRAIN_N \
  --capture24_test_n $TEST_N \
  --capture24_keep_ratio $KEEP_RATIO \
  --downsample_factor $DOWNSAMPLE \
  --capture24_label_col "$LABEL_COL" \
  --stage 2 \
  --batch_size $BATCH_SIZE \
  --llama_name $LLAMA_NAME \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --trainable_modules $TRAIN_trainable_modules \
  --num_workers 10 \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done CAPTURE24."

fi

echo "========================================================"
echo "All tasks finished successfully!"
echo "Global Log Root: $GLOBAL_LOG_ROOT"
echo "========================================================"


#python D:\fuy\MyCode/SensorLLMLib/send_email.py


#shutdown -h now
