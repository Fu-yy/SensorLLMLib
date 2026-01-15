#!/bin/bash
#set -e  # 遇到错误立即停止。如果希望忽略错误继续跑下一个，请注释掉这一行
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
if false;then

for _ in 1;do


TARGET_DIR="/root/autodl-tmp/SensorLLMLib/pretrain_ckpts"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi

TARGET_DIR="/root/autodl-tmp/SensorLLMLib/checkpoints"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi

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


LLAMA_NAME="/root/autodl-tmp/Llama-3.2-1B"
#LLAMA_NAME="/root/autodl-tmp/Llama-3-8B"
PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"

# ========================================================
# 1. UCIHAR
# ========================================================
echo "[1/5] Running UCIHAR..."
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
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
model_name=SensorLLMFuy_01_01_expert

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
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/USC-HAD/USC-HAD"
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
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/MHEALTHDATASET"
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


echo "========================================================"
echo "All tasks finished successfully!"
echo "Global Log Root: $GLOBAL_LOG_ROOT"
echo "========================================================"


done


for _ in 1;do


TARGET_DIR="/root/autodl-tmp/SensorLLMLib/pretrain_ckpts"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi

TARGET_DIR="/root/autodl-tmp/SensorLLMLib/checkpoints"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi
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


LLAMA_NAME="/root/autodl-tmp/Llama-3.2-1B"
#LLAMA_NAME="/root/autodl-tmp/Llama-3-8B"
PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"

# ========================================================
# 1. UCIHAR
# ========================================================
echo "[1/5] Running UCIHAR..."
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
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
model_name=SensorLLMFuy_01_01_identity

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
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/USC-HAD/USC-HAD"
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
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/MHEALTHDATASET"
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


echo "========================================================"
echo "All tasks finished successfully!"
echo "Global Log Root: $GLOBAL_LOG_ROOT"
echo "========================================================"


done

for _ in 1;do



TARGET_DIR="/root/autodl-tmp/SensorLLMLib/pretrain_ckpts"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi

TARGET_DIR="/root/autodl-tmp/SensorLLMLib/checkpoints"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi
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


LLAMA_NAME="/root/autodl-tmp/Llama-3.2-1B"
#LLAMA_NAME="/root/autodl-tmp/Llama-3-8B"
PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"

# ========================================================
# 1. UCIHAR
# ========================================================
echo "[1/5] Running UCIHAR..."
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
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
model_name=SensorLLMFuy_01_01_null

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
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/USC-HAD/USC-HAD"
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
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/MHEALTHDATASET"
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


echo "========================================================"
echo "All tasks finished successfully!"
echo "Global Log Root: $GLOBAL_LOG_ROOT"
echo "========================================================"


done


for _ in 1;do


TARGET_DIR="/root/autodl-tmp/SensorLLMLib/pretrain_ckpts"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi

TARGET_DIR="/root/autodl-tmp/SensorLLMLib/checkpoints"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi
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


LLAMA_NAME="/root/autodl-tmp/Llama-3.2-1B"
#LLAMA_NAME="/root/autodl-tmp/Llama-3-8B"
PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"

# ========================================================
# 1. UCIHAR
# ========================================================
echo "[1/5] Running UCIHAR..."
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
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
model_name=SensorLLMFuy_01_02

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
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/USC-HAD/USC-HAD"
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
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/MHEALTHDATASET"
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


echo "========================================================"
echo "All tasks finished successfully!"
echo "Global Log Root: $GLOBAL_LOG_ROOT"
echo "========================================================"


done


for _ in 1;do


TARGET_DIR="/root/autodl-tmp/SensorLLMLib/pretrain_ckpts"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi

TARGET_DIR="/root/autodl-tmp/SensorLLMLib/checkpoints"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi
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


LLAMA_NAME="/root/autodl-tmp/Llama-3.2-1B"
#LLAMA_NAME="/root/autodl-tmp/Llama-3-8B"
PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"

# ========================================================
# 1. UCIHAR
# ========================================================
echo "[1/5] Running UCIHAR..."
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
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
model_name=SensorLLMFuy_01_03

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
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/USC-HAD/USC-HAD"
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
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/MHEALTHDATASET"
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


echo "========================================================"
echo "All tasks finished successfully!"
echo "Global Log Root: $GLOBAL_LOG_ROOT"
echo "========================================================"


done


for _ in 1;do


TARGET_DIR="/root/autodl-tmp/SensorLLMLib/pretrain_ckpts"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi

TARGET_DIR="/root/autodl-tmp/SensorLLMLib/checkpoints"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi
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


LLAMA_NAME="/root/autodl-tmp/Llama-3.2-1B"
#LLAMA_NAME="/root/autodl-tmp/Llama-3-8B"
PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"

# ========================================================
# 1. UCIHAR
# ========================================================
echo "[1/5] Running UCIHAR..."
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
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
model_name=SensorLLMFuy_01_04

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
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/USC-HAD/USC-HAD"
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
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/MHEALTHDATASET"
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


echo "========================================================"
echo "All tasks finished successfully!"
echo "Global Log Root: $GLOBAL_LOG_ROOT"
echo "========================================================"


done

fi
for _ in 1;do


TARGET_DIR="/root/autodl-tmp/SensorLLMLib/pretrain_ckpts"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi

TARGET_DIR="/root/autodl-tmp/SensorLLMLib/checkpoints"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi

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


#LLAMA_NAME="/root/autodl-tmp/Llama-3.2-1B"
LLAMA_NAME="/root/autodl-tmp/Llama-3-8B"
PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"

# ========================================================
# 1. UCIHAR
# ========================================================
echo "[1/5] Running UCIHAR..."
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
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
BATCH_SIZE=2
LR=0.001
EPOCHS=8
model_name=SensorLLMFuy_01_01_expert

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




TARGET_DIR="/root/autodl-tmp/SensorLLMLib/pretrain_ckpts"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi

TARGET_DIR="/root/autodl-tmp/SensorLLMLib/checkpoints"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi
# ========================================================
# 2. USC-HAD
# ========================================================
echo "[2/5] Running USC-HAD..."
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/USC-HAD/USC-HAD"
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
BATCH_SIZE=2 # 注意这里变了
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



TARGET_DIR="/root/autodl-tmp/SensorLLMLib/pretrain_ckpts"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi

TARGET_DIR="/root/autodl-tmp/SensorLLMLib/checkpoints"

# 1. 检查变量是否为空
# 2. 检查目录是否存在
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"/*
    echo "清理完成"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi
# ========================================================
# 3. MHEALTH
# ========================================================
echo "[3/5] Running MHEALTH..."
DATA_ROOT="/root/autodl-tmp/SensorLLMLib/datasets/MHEALTHDATASET"
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
BATCH_SIZE=2

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


echo "========================================================"
echo "All tasks finished successfully!"
echo "Global Log Root: $GLOBAL_LOG_ROOT"
echo "========================================================"


done



python /root/autodl-tmp/SensorLLMLib/send_email.py


shutdown -h now
