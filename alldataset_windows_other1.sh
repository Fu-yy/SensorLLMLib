#!/bin/bash
#set -e  # 遇到错误立即停止。如果希望忽略错误继续跑下一个，请注释掉这一行

# ===== 全局设置 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


LLAMA_NAME="D:\fuy\MyCode/Llama-3.2-1B"


for i in 1;do



# 生成一个全局时间标签，这样这一次批量运行的所有日志都在同一个大目录下
GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG_ROOT="./run_log/batch_${GLOBAL_TIME_TAG}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting Batch Training Run: $GLOBAL_TIME_TAG"
echo "Logs will be saved to: $GLOBAL_LOG_ROOT"
echo "========================================================"

model_name=SensorLLMFuy_test_nollm_contri
PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"


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






PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"

# ========================================================
# 1. WISDM
# ========================================================
echo "[1/4] Running WISDM..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets\WISDM_ar_latest\WISDM_ar_v1.1"   # <- 改成你的路径(文件夹内有 WISDM_ar_v1.1_raw.txt) 或直接指向 raw.txt
DATA_KEY="wisdm"
DATA_NAME="WISDM"
RUN_ID="${GLOBAL_TIME_TAG}_wisdm_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/wisdm"
mkdir -p "$LOG_DIR"

# Settings (WISDM ~20Hz)
ALIGN_W_MAX=200
SEQ_LEN=80
PATCH_LEN=40
BATCH_SIZE=64
LR=0.001
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id WISDM --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  --test_users 33,34,35,36 \
  --val_users 5,13,17,19,27,31 \
  --wisdm_norm none \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id WISDM --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  --test_users 33,34,35,36 \
  --val_users 5,13,17,19,27,31 \
  --wisdm_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done WISDM."


# ========================================================
# 2. HHAR_1user
# ========================================================
echo "[2/4] Running HHAR_1user..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets\heterogeneity+activity+recognition\Activity recognition exp\Activity recognition exp"   # <- 改成你的路径(含 Phones_*.csv / Watch_*.csv)
DATA_KEY="hhar"
DATA_NAME="HHAR_1user"
RUN_ID="${GLOBAL_TIME_TAG}_hhar_1user_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/hhar_1user"
mkdir -p "$LOG_DIR"

# Settings (HHAR)
ALIGN_W_MAX=256
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_1user --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_1user --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done HHAR_1user."


# ========================================================
# 3. HHAR_cross_user
# ========================================================
echo "[3/4] Running HHAR_cross_user..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets\heterogeneity+activity+recognition\Activity recognition exp\Activity recognition exp"   # <- 同上
DATA_KEY="hhar"
DATA_NAME="HHAR_cross_user"
RUN_ID="${GLOBAL_TIME_TAG}_hhar_cross_user_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/hhar_cross_user"
mkdir -p "$LOG_DIR"

# Settings (HHAR)
ALIGN_W_MAX=256
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_cross_user --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  --val_ratio 0.1 --test_ratio 0.2 \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_cross_user --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  --val_ratio 0.1 --test_ratio 0.2 \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done HHAR_cross_user."


# ========================================================
# 4. MotionSense
# ========================================================
echo "[4/4] Running MotionSense..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets\motion-sense-master\motion-sense-master\data"  # <- 改成你的路径
DATA_KEY="motionsense"
DATA_NAME="MotionSense"
RUN_ID="${GLOBAL_TIME_TAG}_motionsense_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/motionsense"
mkdir -p "$LOG_DIR"

# Settings (MotionSense ~50Hz)
ALIGN_W_MAX=256
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
EPOCHS=8



# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MotionSense --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  --test_users 19,20,21,22,23,24 \
  --val_users 13,14,15,16,17,18 \
  --motionsense_feature_set A12 \
  --motionsense_combine_grav_acc 0 \
  --motionsense_norm none \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MotionSense --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  --test_users 19,20,21,22,23,24 \
  --val_users 13,14,15,16,17,18 \
  --motionsense_feature_set A12 \
  --motionsense_combine_grav_acc 0 \
  --motionsense_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done MotionSense."

echo "========================================================"
echo "ALL DONE. Logs at: $GLOBAL_LOG_ROOT"
echo "========================================================"


done



###################################### new model ###############################################
#model_name=SensorLLMFuy_test_nollm_mae

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


LLAMA_NAME="D:\fuy\MyCode/Llama-3.2-1B"



for i in 1;do


# 生成一个全局时间标签，这样这一次批量运行的所有日志都在同一个大目录下
GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG_ROOT="./run_log/batch_${GLOBAL_TIME_TAG}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting Batch Training Run: $GLOBAL_TIME_TAG"
echo "Logs will be saved to: $GLOBAL_LOG_ROOT"
echo "========================================================"

model_name=SensorLLMFuy_test_nollm_mae
PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"


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






PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"

# ========================================================
# 1. WISDM
# ========================================================
echo "[1/4] Running WISDM..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets\WISDM_ar_latest\WISDM_ar_v1.1"   # <- 改成你的路径(文件夹内有 WISDM_ar_v1.1_raw.txt) 或直接指向 raw.txt
DATA_KEY="wisdm"
DATA_NAME="WISDM"
RUN_ID="${GLOBAL_TIME_TAG}_wisdm_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/wisdm"
mkdir -p "$LOG_DIR"

# Settings (WISDM ~20Hz)
ALIGN_W_MAX=200
SEQ_LEN=80
PATCH_LEN=40
BATCH_SIZE=64
LR=0.001
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id WISDM --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  --test_users 33,34,35,36 \
  --val_users 5,13,17,19,27,31 \
  --wisdm_norm none \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id WISDM --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  --test_users 33,34,35,36 \
  --val_users 5,13,17,19,27,31 \
  --wisdm_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done WISDM."


# ========================================================
# 2. HHAR_1user
# ========================================================
echo "[2/4] Running HHAR_1user..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets\heterogeneity+activity+recognition\Activity recognition exp\Activity recognition exp"   # <- 改成你的路径(含 Phones_*.csv / Watch_*.csv)
DATA_KEY="hhar"
DATA_NAME="HHAR_1user"
RUN_ID="${GLOBAL_TIME_TAG}_hhar_1user_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/hhar_1user"
mkdir -p "$LOG_DIR"

# Settings (HHAR)
ALIGN_W_MAX=256
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_1user --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_1user --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done HHAR_1user."


# ========================================================
# 3. HHAR_cross_user
# ========================================================
echo "[3/4] Running HHAR_cross_user..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets\heterogeneity+activity+recognition\Activity recognition exp\Activity recognition exp"   # <- 同上
DATA_KEY="hhar"
DATA_NAME="HHAR_cross_user"
RUN_ID="${GLOBAL_TIME_TAG}_hhar_cross_user_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/hhar_cross_user"
mkdir -p "$LOG_DIR"

# Settings (HHAR)
ALIGN_W_MAX=256
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_cross_user --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  --val_ratio 0.1 --test_ratio 0.2 \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_cross_user --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  --val_ratio 0.1 --test_ratio 0.2 \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done HHAR_cross_user."


# ========================================================
# 4. MotionSense
# ========================================================
echo "[4/4] Running MotionSense..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets\motion-sense-master\motion-sense-master\data"  # <- 改成你的路径
DATA_KEY="motionsense"
DATA_NAME="MotionSense"
RUN_ID="${GLOBAL_TIME_TAG}_motionsense_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/motionsense"
mkdir -p "$LOG_DIR"

# Settings (MotionSense ~50Hz)
ALIGN_W_MAX=256
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MotionSense --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  --test_users 19,20,21,22,23,24 \
  --val_users 13,14,15,16,17,18 \
  --motionsense_feature_set A12 \
  --motionsense_combine_grav_acc 0 \
  --motionsense_norm none \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MotionSense --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 \
  --test_users 19,20,21,22,23,24 \
  --val_users 13,14,15,16,17,18 \
  --motionsense_feature_set A12 \
  --motionsense_combine_grav_acc 0 \
  --motionsense_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done MotionSense."

echo "========================================================"
echo "ALL DONE. Logs at: $GLOBAL_LOG_ROOT"
echo "========================================================"


done

if false;then
###################################### new model ###############################################
#model_name=iTransformer

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


LLAMA_NAME="D:\fuy\MyCode/Llama-3.2-1B"



for i in 1;do

# 生成一个全局时间标签，这样这一次批量运行的所有日志都在同一个大目录下
GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG_ROOT="./run_log/batch_${GLOBAL_TIME_TAG}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting Batch Training Run: $GLOBAL_TIME_TAG"
echo "Logs will be saved to: $GLOBAL_LOG_ROOT"
echo "========================================================"

model_name=iTransformer
PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"
DMODEL=128
DFF=256

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

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 --d_model $DMODEL --d_ff $DFF \
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


# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id USCHAD --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 --d_model $DMODEL --d_ff $DFF \
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


# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MHealth --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 --d_model $DMODEL --d_ff $DFF \
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

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id PAMAP2 --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 --d_model $DMODEL --d_ff $DFF \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done PAMAP50 (50Hz)."







# ========================================================
# 1. WISDM
# ========================================================
echo "[1/4] Running WISDM..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets\WISDM_ar_latest\WISDM_ar_v1.1"   # <- 改成你的路径(文件夹内有 WISDM_ar_v1.1_raw.txt) 或直接指向 raw.txt
DATA_KEY="wisdm"
DATA_NAME="WISDM"
RUN_ID="${GLOBAL_TIME_TAG}_wisdm_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/wisdm"
mkdir -p "$LOG_DIR"

# Settings (WISDM ~20Hz)
ALIGN_W_MAX=200
SEQ_LEN=80
PATCH_LEN=40
BATCH_SIZE=64
LR=0.001
EPOCHS=8

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id WISDM --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 --d_model $DMODEL --d_ff $DFF \
  --test_users 33,34,35,36 \
  --val_users 5,13,17,19,27,31 \
  --wisdm_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done WISDM."


# ========================================================
# 2. HHAR_1user
# ========================================================
echo "[2/4] Running HHAR_1user..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets\heterogeneity+activity+recognition\Activity recognition exp\Activity recognition exp"   # <- 改成你的路径(含 Phones_*.csv / Watch_*.csv)
DATA_KEY="hhar"
DATA_NAME="HHAR_1user"
RUN_ID="${GLOBAL_TIME_TAG}_hhar_1user_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/hhar_1user"
mkdir -p "$LOG_DIR"

# Settings (HHAR)
ALIGN_W_MAX=256
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
EPOCHS=8


# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_1user --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 --d_model $DMODEL --d_ff $DFF \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done HHAR_1user."


# ========================================================
# 3. HHAR_cross_user
# ========================================================
echo "[3/4] Running HHAR_cross_user..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets\heterogeneity+activity+recognition\Activity recognition exp\Activity recognition exp"   # <- 同上
DATA_KEY="hhar"
DATA_NAME="HHAR_cross_user"
RUN_ID="${GLOBAL_TIME_TAG}_hhar_cross_user_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/hhar_cross_user"
mkdir -p "$LOG_DIR"

# Settings (HHAR)
ALIGN_W_MAX=256
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
EPOCHS=8


# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_cross_user --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 --d_model $DMODEL --d_ff $DFF \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  --val_ratio 0.1 --test_ratio 0.2 \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done HHAR_cross_user."


# ========================================================
# 4. MotionSense
# ========================================================
echo "[4/4] Running MotionSense..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets\motion-sense-master\motion-sense-master\data"  # <- 改成你的路径
DATA_KEY="motionsense"
DATA_NAME="MotionSense"
RUN_ID="${GLOBAL_TIME_TAG}_motionsense_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/motionsense"
mkdir -p "$LOG_DIR"

# Settings (MotionSense ~50Hz)
ALIGN_W_MAX=256
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
EPOCHS=8


# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MotionSense --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 --d_model $DMODEL --d_ff $DFF \
  --test_users 19,20,21,22,23,24 \
  --val_users 13,14,15,16,17,18 \
  --motionsense_feature_set A12 \
  --motionsense_combine_grav_acc 0 \
  --motionsense_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done MotionSense."

echo "========================================================"
echo "ALL DONE. Logs at: $GLOBAL_LOG_ROOT"
echo "========================================================"


done





###################################### new model ###############################################
#model_name=iTransformer

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


LLAMA_NAME="D:\fuy\MyCode/Llama-3.2-1B"



for i in 1;do

# 生成一个全局时间标签，这样这一次批量运行的所有日志都在同一个大目录下
GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG_ROOT="./run_log/batch_${GLOBAL_TIME_TAG}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting Batch Training Run: $GLOBAL_TIME_TAG"
echo "Logs will be saved to: $GLOBAL_LOG_ROOT"
echo "========================================================"

model_name=TimesNet
PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"
DMODEL=32
DFF=32

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

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 --d_model $DMODEL --d_ff $DFF \
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


# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id USCHAD --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 --d_model $DMODEL --d_ff $DFF \
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


# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MHealth --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 --d_model $DMODEL --d_ff $DFF \
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

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id PAMAP2 --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 --d_model $DMODEL --d_ff $DFF \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done PAMAP50 (50Hz)."







# ========================================================
# 1. WISDM
# ========================================================
echo "[1/4] Running WISDM..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets\WISDM_ar_latest\WISDM_ar_v1.1"   # <- 改成你的路径(文件夹内有 WISDM_ar_v1.1_raw.txt) 或直接指向 raw.txt
DATA_KEY="wisdm"
DATA_NAME="WISDM"
RUN_ID="${GLOBAL_TIME_TAG}_wisdm_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/wisdm"
mkdir -p "$LOG_DIR"

# Settings (WISDM ~20Hz)
ALIGN_W_MAX=200
SEQ_LEN=80
PATCH_LEN=40
BATCH_SIZE=64
LR=0.001
EPOCHS=8

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id WISDM --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 --d_model $DMODEL --d_ff $DFF \
  --test_users 33,34,35,36 \
  --val_users 5,13,17,19,27,31 \
  --wisdm_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done WISDM."


# ========================================================
# 2. HHAR_1user
# ========================================================
echo "[2/4] Running HHAR_1user..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets\heterogeneity+activity+recognition\Activity recognition exp\Activity recognition exp"   # <- 改成你的路径(含 Phones_*.csv / Watch_*.csv)
DATA_KEY="hhar"
DATA_NAME="HHAR_1user"
RUN_ID="${GLOBAL_TIME_TAG}_hhar_1user_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/hhar_1user"
mkdir -p "$LOG_DIR"

# Settings (HHAR)
ALIGN_W_MAX=256
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
EPOCHS=8


# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_1user --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 --d_model $DMODEL --d_ff $DFF \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done HHAR_1user."


# ========================================================
# 3. HHAR_cross_user
# ========================================================
echo "[3/4] Running HHAR_cross_user..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets\heterogeneity+activity+recognition\Activity recognition exp\Activity recognition exp"   # <- 同上
DATA_KEY="hhar"
DATA_NAME="HHAR_cross_user"
RUN_ID="${GLOBAL_TIME_TAG}_hhar_cross_user_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/hhar_cross_user"
mkdir -p "$LOG_DIR"

# Settings (HHAR)
ALIGN_W_MAX=256
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
EPOCHS=8


# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_cross_user --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 --d_model $DMODEL --d_ff $DFF \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  --val_ratio 0.1 --test_ratio 0.2 \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done HHAR_cross_user."


# ========================================================
# 4. MotionSense
# ========================================================
echo "[4/4] Running MotionSense..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets\motion-sense-master\motion-sense-master\data"  # <- 改成你的路径
DATA_KEY="motionsense"
DATA_NAME="MotionSense"
RUN_ID="${GLOBAL_TIME_TAG}_motionsense_sensorllm"
LOG_DIR="$GLOBAL_LOG_ROOT/motionsense"
mkdir -p "$LOG_DIR"

# Settings (MotionSense ~50Hz)
ALIGN_W_MAX=256
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
EPOCHS=8


# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MotionSense --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 10 --d_model $DMODEL --d_ff $DFF \
  --test_users 19,20,21,22,23,24 \
  --val_users 13,14,15,16,17,18 \
  --motionsense_feature_set A12 \
  --motionsense_combine_grav_acc 0 \
  --motionsense_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done MotionSense."

echo "========================================================"
echo "ALL DONE. Logs at: $GLOBAL_LOG_ROOT"
echo "========================================================"


done
fi

#python D:\fuy\MyCode/SensorLLMLib/send_email.py


#shutdown -h now
