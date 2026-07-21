#!/bin/bash
#set -e

# ===== 全局设置 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LLAMA_NAME="/root/autodl-tmp/Llama-3.2-1B"
LLAMA_NAME="D:\fuy\MyCode/Llama-3.2-1B"

TS_BACKBONE_YAML="ts_backbone.yaml"
TS_BACKBONE_YAML="ts_backbone_linux.yaml"

########################################################
############# CE-only: random mask, no LLM distill #####
########################################################

MASK_RATE=0.4
use_hard_label=0
mask_mode="random"
LAMBDA_DISTILL=0

VQVAEPATH="qua_recon_path"

GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG_ROOT="./run_log/batch_${GLOBAL_TIME_TAG}_ce_only_random_mask_${MASK_RATE}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting CE-only Batch Training Run: $GLOBAL_TIME_TAG"
echo "Stage1: random mask + primitive CE only"
echo "lambda_distill=$LAMBDA_DISTILL"
echo "mask_mode=$mask_mode"
echo "mask_rate=$MASK_RATE"
echo "Logs will be saved to: $GLOBAL_LOG_ROOT"
echo "========================================================"

model_name=SensorLLMFuy_test_withllm_mae_vqvae

# 对应新模型结构：
# Stage1 训练 student，包括 patch_embed / transformer / vocab_head
# Stage2 训练 student + attention pooling + classifier
PRETRAIN_trainable_modules="student"
TRAIN_trainable_modules="student,pool_query,pool_attn,classifier"


for i in 1; do

# ========================================================
# 清理缓存函数
# ========================================================
clean_cache() {
    TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v2"
    if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
        find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
        echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
    else
        echo "错误：目录不存在或变量为空，跳过删除"
    fi
}

# ========================================================
# 1. UCIHAR
# ========================================================
clean_cache

echo "[1/7] Running UCIHAR..."
DATA_ROOT="/root/autodl-tmp/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"

DATA_KEY="ucihar"
DATA_NAME="UCIHAR"
RUN_ID="${GLOBAL_TIME_TAG}_ucihar_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/ucihar"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=128
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=32
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1: CE-only primitive prediction
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2: normal fine-tuning
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done UCIHAR."

# ========================================================
# 2. USC-HAD
# ========================================================
clean_cache

echo "[2/7] Running USC-HAD..."
DATA_ROOT="/root/autodl-tmp/datasets/USC-HAD/USC-HAD"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/USC-HAD/USC-HAD"

DATA_KEY="uschad"
DATA_NAME="USCHAD"
RUN_ID="${GLOBAL_TIME_TAG}_uschad_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/uschad"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=200
SEQ_LEN=200
PATCH_LEN=100
BATCH_SIZE=16
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8
TEST_SUBJECTS="subject13,subject14"

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id USCHAD --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id USCHAD --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done USC-HAD."

# ========================================================
# 3. MHEALTH
# ========================================================
clean_cache

echo "[3/7] Running MHEALTH..."
DATA_ROOT="/root/autodl-tmp/datasets/MHEALTHDATASET"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/MHEALTHDATASET"

DATA_KEY="mhealth"
DATA_NAME="MHealth"
RUN_ID="${GLOBAL_TIME_TAG}_mhealth_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/mhealth"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=100
SEQ_LEN=100
PATCH_LEN=50
TEST_SUBJECTS="subject1,subject3,subject6"
BATCH_SIZE=16
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MHealth --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MHealth --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done MHEALTH."

# ========================================================
# 4. PAMAP2 50Hz
# ========================================================
clean_cache

echo "[4/7] Running PAMAP2 (50Hz)..."
DATA_ROOT="/root/autodl-tmp/datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"

DATA_KEY="pamap50"
DATA_NAME="PAMAP50"
RUN_ID="${GLOBAL_TIME_TAG}_pamap50_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/pamap50_50hz"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=100
SEQ_LEN=100
PATCH_LEN=50
PAMAP_VARIANT="pamap50"
TEST_SUBJECTS="subject105,subject106"
BATCH_SIZE=32
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id PAMAP2 --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id PAMAP2 --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done PAMAP50."

# ========================================================
# 5. WISDM
# ========================================================
clean_cache

echo "[5/7] Running WISDM..."
DATA_ROOT="/root/autodl-tmp/datasets/WISDM_ar_latest/WISDM_ar_v1.1"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/WISDM_ar_latest/WISDM_ar_v1.1"   # <- 改成你的路径(文件夹内有 WISDM_ar_v1.1_raw.txt) 或直接指向 raw.txt

DATA_KEY="wisdm"
DATA_NAME="WISDM"
RUN_ID="${GLOBAL_TIME_TAG}_wisdm_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/wisdm"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=80
SEQ_LEN=80
PATCH_LEN=40
BATCH_SIZE=64
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id WISDM --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  --test_users 33,34,35,36 \
  --val_users 5,13,17,19,27,31 \
  --wisdm_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done WISDM."

# ========================================================
# 6. HHAR_1user
# ========================================================
clean_cache

echo "[6/7] Running HHAR_1user..."
DATA_ROOT="/root/autodl-tmp/datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"   # <- 改成你的路径(含 Phones_*.csv / Watch_*.csv)

DATA_KEY="hhar"
DATA_NAME="HHAR_1user"
RUN_ID="${GLOBAL_TIME_TAG}_hhar_1user_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/hhar_1user"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=128
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_1user --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done HHAR_1user."

# ========================================================
# 7. MotionSense
# ========================================================
clean_cache

echo "[7/7] Running MotionSense..."
DATA_ROOT="/root/autodl-tmp/datasets/motion-sense-master/motion-sense-master/data"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/motion-sense-master/motion-sense-master/data"  # <- 改成你的路径

DATA_KEY="motionsense"
DATA_NAME="MotionSense"
RUN_ID="${GLOBAL_TIME_TAG}_motionsense_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/motionsense"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=128
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MotionSense --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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





################################################################
################################################################
################################################################
################################################################
################################################################


GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG_ROOT="./run_log/batch_${GLOBAL_TIME_TAG}_ce_only_random_mask_${MASK_RATE}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting CE-only Batch Training Run: $GLOBAL_TIME_TAG"
echo "Stage1: random mask + primitive CE only"
echo "lambda_distill=$LAMBDA_DISTILL"
echo "mask_mode=$mask_mode"
echo "mask_rate=$MASK_RATE"
echo "Logs will be saved to: $GLOBAL_LOG_ROOT"
echo "========================================================"

model_name=SensorLLMFuy_test_withllm_mae_vqvae

# 对应新模型结构：
# Stage1 训练 student，包括 patch_embed / transformer / vocab_head
# Stage2 训练 student + attention pooling + classifier
PRETRAIN_trainable_modules="student"
TRAIN_trainable_modules="student,pool_query,pool_attn,classifier"

for i in 1; do

# ========================================================
# 清理缓存函数
# ========================================================
clean_cache() {
    TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v2"
    if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
        find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
        echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
    else
        echo "错误：目录不存在或变量为空，跳过删除"
    fi
}

# ========================================================
# 1. UCIHAR
# ========================================================
clean_cache

echo "[1/7] Running UCIHAR..."
DATA_ROOT="/root/autodl-tmp/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"

DATA_KEY="ucihar"
DATA_NAME="UCIHAR"
RUN_ID="${GLOBAL_TIME_TAG}_ucihar_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/ucihar"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=128
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=32
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1: CE-only primitive prediction
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2: normal fine-tuning
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done UCIHAR."

# ========================================================
# 2. USC-HAD
# ========================================================
clean_cache

echo "[2/7] Running USC-HAD..."
DATA_ROOT="/root/autodl-tmp/datasets/USC-HAD/USC-HAD"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/USC-HAD/USC-HAD"

DATA_KEY="uschad"
DATA_NAME="USCHAD"
RUN_ID="${GLOBAL_TIME_TAG}_uschad_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/uschad"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=200
SEQ_LEN=200
PATCH_LEN=100
BATCH_SIZE=16
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8
TEST_SUBJECTS="subject13,subject14"

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id USCHAD --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id USCHAD --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done USC-HAD."

# ========================================================
# 3. MHEALTH
# ========================================================
clean_cache

echo "[3/7] Running MHEALTH..."
DATA_ROOT="/root/autodl-tmp/datasets/MHEALTHDATASET"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/MHEALTHDATASET"

DATA_KEY="mhealth"
DATA_NAME="MHealth"
RUN_ID="${GLOBAL_TIME_TAG}_mhealth_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/mhealth"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=100
SEQ_LEN=100
PATCH_LEN=50
TEST_SUBJECTS="subject1,subject3,subject6"
BATCH_SIZE=16
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MHealth --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MHealth --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done MHEALTH."

# ========================================================
# 4. PAMAP2 50Hz
# ========================================================
clean_cache

echo "[4/7] Running PAMAP2 (50Hz)..."
DATA_ROOT="/root/autodl-tmp/datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"

DATA_KEY="pamap50"
DATA_NAME="PAMAP50"
RUN_ID="${GLOBAL_TIME_TAG}_pamap50_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/pamap50_50hz"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=100
SEQ_LEN=100
PATCH_LEN=50
PAMAP_VARIANT="pamap50"
TEST_SUBJECTS="subject105,subject106"
BATCH_SIZE=32
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id PAMAP2 --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id PAMAP2 --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done PAMAP50."

# ========================================================
# 5. WISDM
# ========================================================
clean_cache

echo "[5/7] Running WISDM..."
DATA_ROOT="/root/autodl-tmp/datasets/WISDM_ar_latest/WISDM_ar_v1.1"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/WISDM_ar_latest/WISDM_ar_v1.1"   # <- 改成你的路径(文件夹内有 WISDM_ar_v1.1_raw.txt) 或直接指向 raw.txt

DATA_KEY="wisdm"
DATA_NAME="WISDM"
RUN_ID="${GLOBAL_TIME_TAG}_wisdm_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/wisdm"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=80
SEQ_LEN=80
PATCH_LEN=40
BATCH_SIZE=64
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id WISDM --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  --test_users 33,34,35,36 \
  --val_users 5,13,17,19,27,31 \
  --wisdm_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done WISDM."

# ========================================================
# 6. HHAR_1user
# ========================================================
clean_cache

echo "[6/7] Running HHAR_1user..."
DATA_ROOT="/root/autodl-tmp/datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"   # <- 改成你的路径(含 Phones_*.csv / Watch_*.csv)

DATA_KEY="hhar"
DATA_NAME="HHAR_1user"
RUN_ID="${GLOBAL_TIME_TAG}_hhar_1user_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/hhar_1user"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=128
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_1user --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done HHAR_1user."

# ========================================================
# 7. MotionSense
# ========================================================
clean_cache

echo "[7/7] Running MotionSense..."
DATA_ROOT="/root/autodl-tmp/datasets/motion-sense-master/motion-sense-master/data"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/motion-sense-master/motion-sense-master/data"  # <- 改成你的路径

DATA_KEY="motionsense"
DATA_NAME="MotionSense"
RUN_ID="${GLOBAL_TIME_TAG}_motionsense_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/motionsense"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=128
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MotionSense --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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


################################################################
################################################################
################################################################
################################################################
################################################################


GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG_ROOT="./run_log/batch_${GLOBAL_TIME_TAG}_ce_only_random_mask_${MASK_RATE}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting CE-only Batch Training Run: $GLOBAL_TIME_TAG"
echo "Stage1: random mask + primitive CE only"
echo "lambda_distill=$LAMBDA_DISTILL"
echo "mask_mode=$mask_mode"
echo "mask_rate=$MASK_RATE"
echo "Logs will be saved to: $GLOBAL_LOG_ROOT"
echo "========================================================"

model_name=SensorLLMFuy_test_withllm_mae_vqvae

# 对应新模型结构：
# Stage1 训练 student，包括 patch_embed / transformer / vocab_head
# Stage2 训练 student + attention pooling + classifier
PRETRAIN_trainable_modules="student"
TRAIN_trainable_modules="student,pool_query,pool_attn,classifier"


for i in 1; do

# ========================================================
# 清理缓存函数
# ========================================================
clean_cache() {
    TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v2"
    if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
        find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
        echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
    else
        echo "错误：目录不存在或变量为空，跳过删除"
    fi
}

# ========================================================
# 1. UCIHAR
# ========================================================
clean_cache

echo "[1/7] Running UCIHAR..."
DATA_ROOT="/root/autodl-tmp/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"

DATA_KEY="ucihar"
DATA_NAME="UCIHAR"
RUN_ID="${GLOBAL_TIME_TAG}_ucihar_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/ucihar"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=128
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=32
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1: CE-only primitive prediction
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2: normal fine-tuning
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done UCIHAR."

# ========================================================
# 2. USC-HAD
# ========================================================
clean_cache

echo "[2/7] Running USC-HAD..."
DATA_ROOT="/root/autodl-tmp/datasets/USC-HAD/USC-HAD"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/USC-HAD/USC-HAD"

DATA_KEY="uschad"
DATA_NAME="USCHAD"
RUN_ID="${GLOBAL_TIME_TAG}_uschad_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/uschad"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=200
SEQ_LEN=200
PATCH_LEN=100
BATCH_SIZE=16
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8
TEST_SUBJECTS="subject13,subject14"

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id USCHAD --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id USCHAD --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done USC-HAD."

# ========================================================
# 3. MHEALTH
# ========================================================
clean_cache

echo "[3/7] Running MHEALTH..."
DATA_ROOT="/root/autodl-tmp/datasets/MHEALTHDATASET"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/MHEALTHDATASET"

DATA_KEY="mhealth"
DATA_NAME="MHealth"
RUN_ID="${GLOBAL_TIME_TAG}_mhealth_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/mhealth"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=100
SEQ_LEN=100
PATCH_LEN=50
TEST_SUBJECTS="subject1,subject3,subject6"
BATCH_SIZE=16
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MHealth --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MHealth --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done MHEALTH."

# ========================================================
# 4. PAMAP2 50Hz
# ========================================================
clean_cache

echo "[4/7] Running PAMAP2 (50Hz)..."
DATA_ROOT="/root/autodl-tmp/datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"

DATA_KEY="pamap50"
DATA_NAME="PAMAP50"
RUN_ID="${GLOBAL_TIME_TAG}_pamap50_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/pamap50_50hz"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=100
SEQ_LEN=100
PATCH_LEN=50
PAMAP_VARIANT="pamap50"
TEST_SUBJECTS="subject105,subject106"
BATCH_SIZE=32
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id PAMAP2 --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id PAMAP2 --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done PAMAP50."

# ========================================================
# 5. WISDM
# ========================================================
clean_cache

echo "[5/7] Running WISDM..."
DATA_ROOT="/root/autodl-tmp/datasets/WISDM_ar_latest/WISDM_ar_v1.1"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/WISDM_ar_latest/WISDM_ar_v1.1"   # <- 改成你的路径(文件夹内有 WISDM_ar_v1.1_raw.txt) 或直接指向 raw.txt

DATA_KEY="wisdm"
DATA_NAME="WISDM"
RUN_ID="${GLOBAL_TIME_TAG}_wisdm_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/wisdm"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=80
SEQ_LEN=80
PATCH_LEN=40
BATCH_SIZE=64
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id WISDM --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  --test_users 33,34,35,36 \
  --val_users 5,13,17,19,27,31 \
  --wisdm_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done WISDM."

# ========================================================
# 6. HHAR_1user
# ========================================================
clean_cache

echo "[6/7] Running HHAR_1user..."
DATA_ROOT="/root/autodl-tmp/datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"   # <- 改成你的路径(含 Phones_*.csv / Watch_*.csv)

DATA_KEY="hhar"
DATA_NAME="HHAR_1user"
RUN_ID="${GLOBAL_TIME_TAG}_hhar_1user_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/hhar_1user"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=128
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_1user --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done HHAR_1user."

# ========================================================
# 7. MotionSense
# ========================================================
clean_cache

echo "[7/7] Running MotionSense..."
DATA_ROOT="/root/autodl-tmp/datasets/motion-sense-master/motion-sense-master/data"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/motion-sense-master/motion-sense-master/data"  # <- 改成你的路径

DATA_KEY="motionsense"
DATA_NAME="MotionSense"
RUN_ID="${GLOBAL_TIME_TAG}_motionsense_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/motionsense"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=128
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MotionSense --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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


################################################################
################################################################
################################################################
################################################################
################################################################


GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG_ROOT="./run_log/batch_${GLOBAL_TIME_TAG}_ce_only_random_mask_${MASK_RATE}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting CE-only Batch Training Run: $GLOBAL_TIME_TAG"
echo "Stage1: random mask + primitive CE only"
echo "lambda_distill=$LAMBDA_DISTILL"
echo "mask_mode=$mask_mode"
echo "mask_rate=$MASK_RATE"
echo "Logs will be saved to: $GLOBAL_LOG_ROOT"
echo "========================================================"

model_name=SensorLLMFuy_test_withllm_mae_vqvae

# 对应新模型结构：
# Stage1 训练 student，包括 patch_embed / transformer / vocab_head
# Stage2 训练 student + attention pooling + classifier
PRETRAIN_trainable_modules="student"
TRAIN_trainable_modules="student,pool_query,pool_attn,classifier"


for i in 1; do

# ========================================================
# 清理缓存函数
# ========================================================
clean_cache() {
    TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v2"
    if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
        find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
        echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
    else
        echo "错误：目录不存在或变量为空，跳过删除"
    fi
}

# ========================================================
# 1. UCIHAR
# ========================================================
clean_cache

echo "[1/7] Running UCIHAR..."
DATA_ROOT="/root/autodl-tmp/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"

DATA_KEY="ucihar"
DATA_NAME="UCIHAR"
RUN_ID="${GLOBAL_TIME_TAG}_ucihar_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/ucihar"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=128
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=32
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1: CE-only primitive prediction
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2: normal fine-tuning
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done UCIHAR."

# ========================================================
# 2. USC-HAD
# ========================================================
clean_cache

echo "[2/7] Running USC-HAD..."
DATA_ROOT="/root/autodl-tmp/datasets/USC-HAD/USC-HAD"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/USC-HAD/USC-HAD"

DATA_KEY="uschad"
DATA_NAME="USCHAD"
RUN_ID="${GLOBAL_TIME_TAG}_uschad_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/uschad"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=200
SEQ_LEN=200
PATCH_LEN=100
BATCH_SIZE=16
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8
TEST_SUBJECTS="subject13,subject14"

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id USCHAD --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id USCHAD --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done USC-HAD."

# ========================================================
# 3. MHEALTH
# ========================================================
clean_cache

echo "[3/7] Running MHEALTH..."
DATA_ROOT="/root/autodl-tmp/datasets/MHEALTHDATASET"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/MHEALTHDATASET"

DATA_KEY="mhealth"
DATA_NAME="MHealth"
RUN_ID="${GLOBAL_TIME_TAG}_mhealth_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/mhealth"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=100
SEQ_LEN=100
PATCH_LEN=50
TEST_SUBJECTS="subject1,subject3,subject6"
BATCH_SIZE=16
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MHealth --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MHealth --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done MHEALTH."

# ========================================================
# 4. PAMAP2 50Hz
# ========================================================
clean_cache

echo "[4/7] Running PAMAP2 (50Hz)..."
DATA_ROOT="/root/autodl-tmp/datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"

DATA_KEY="pamap50"
DATA_NAME="PAMAP50"
RUN_ID="${GLOBAL_TIME_TAG}_pamap50_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/pamap50_50hz"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=100
SEQ_LEN=100
PATCH_LEN=50
PAMAP_VARIANT="pamap50"
TEST_SUBJECTS="subject105,subject106"
BATCH_SIZE=32
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id PAMAP2 --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id PAMAP2 --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done PAMAP50."

# ========================================================
# 5. WISDM
# ========================================================
clean_cache

echo "[5/7] Running WISDM..."
DATA_ROOT="/root/autodl-tmp/datasets/WISDM_ar_latest/WISDM_ar_v1.1"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/WISDM_ar_latest/WISDM_ar_v1.1"   # <- 改成你的路径(文件夹内有 WISDM_ar_v1.1_raw.txt) 或直接指向 raw.txt

DATA_KEY="wisdm"
DATA_NAME="WISDM"
RUN_ID="${GLOBAL_TIME_TAG}_wisdm_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/wisdm"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=80
SEQ_LEN=80
PATCH_LEN=40
BATCH_SIZE=64
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id WISDM --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  --test_users 33,34,35,36 \
  --val_users 5,13,17,19,27,31 \
  --wisdm_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done WISDM."

# ========================================================
# 6. HHAR_1user
# ========================================================
clean_cache

echo "[6/7] Running HHAR_1user..."
DATA_ROOT="/root/autodl-tmp/datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"   # <- 改成你的路径(含 Phones_*.csv / Watch_*.csv)

DATA_KEY="hhar"
DATA_NAME="HHAR_1user"
RUN_ID="${GLOBAL_TIME_TAG}_hhar_1user_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/hhar_1user"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=128
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_1user --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done HHAR_1user."

# ========================================================
# 7. MotionSense
# ========================================================
clean_cache

echo "[7/7] Running MotionSense..."
DATA_ROOT="/root/autodl-tmp/datasets/motion-sense-master/motion-sense-master/data"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/motion-sense-master/motion-sense-master/data"  # <- 改成你的路径

DATA_KEY="motionsense"
DATA_NAME="MotionSense"
RUN_ID="${GLOBAL_TIME_TAG}_motionsense_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/motionsense"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=128
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MotionSense --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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



################################################################
################################################################
################################################################
################################################################
################################################################


GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG_ROOT="./run_log/batch_${GLOBAL_TIME_TAG}_ce_only_random_mask_${MASK_RATE}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting CE-only Batch Training Run: $GLOBAL_TIME_TAG"
echo "Stage1: random mask + primitive CE only"
echo "lambda_distill=$LAMBDA_DISTILL"
echo "mask_mode=$mask_mode"
echo "mask_rate=$MASK_RATE"
echo "Logs will be saved to: $GLOBAL_LOG_ROOT"
echo "========================================================"

model_name=SensorLLMFuy_test_withllm_mae_vqvae

# 对应新模型结构：
# Stage1 训练 student，包括 patch_embed / transformer / vocab_head
# Stage2 训练 student + attention pooling + classifier
PRETRAIN_trainable_modules="student"
TRAIN_trainable_modules="student,pool_query,pool_attn,classifier"


for i in 1; do

# ========================================================
# 清理缓存函数
# ========================================================
clean_cache() {
    TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v2"
    if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
        find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
        echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
    else
        echo "错误：目录不存在或变量为空，跳过删除"
    fi
}

# ========================================================
# 1. UCIHAR
# ========================================================
clean_cache

echo "[1/7] Running UCIHAR..."
DATA_ROOT="/root/autodl-tmp/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"

DATA_KEY="ucihar"
DATA_NAME="UCIHAR"
RUN_ID="${GLOBAL_TIME_TAG}_ucihar_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/ucihar"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=128
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=32
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1: CE-only primitive prediction
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2: normal fine-tuning
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done UCIHAR."

# ========================================================
# 2. USC-HAD
# ========================================================
clean_cache

echo "[2/7] Running USC-HAD..."
DATA_ROOT="/root/autodl-tmp/datasets/USC-HAD/USC-HAD"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/USC-HAD/USC-HAD"

DATA_KEY="uschad"
DATA_NAME="USCHAD"
RUN_ID="${GLOBAL_TIME_TAG}_uschad_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/uschad"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=200
SEQ_LEN=200
PATCH_LEN=100
BATCH_SIZE=16
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8
TEST_SUBJECTS="subject13,subject14"

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id USCHAD --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id USCHAD --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done USC-HAD."

# ========================================================
# 3. MHEALTH
# ========================================================
clean_cache

echo "[3/7] Running MHEALTH..."
DATA_ROOT="/root/autodl-tmp/datasets/MHEALTHDATASET"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/MHEALTHDATASET"

DATA_KEY="mhealth"
DATA_NAME="MHealth"
RUN_ID="${GLOBAL_TIME_TAG}_mhealth_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/mhealth"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=100
SEQ_LEN=100
PATCH_LEN=50
TEST_SUBJECTS="subject1,subject3,subject6"
BATCH_SIZE=16
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MHealth --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MHealth --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done MHEALTH."

# ========================================================
# 4. PAMAP2 50Hz
# ========================================================
clean_cache

echo "[4/7] Running PAMAP2 (50Hz)..."
DATA_ROOT="/root/autodl-tmp/datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"

DATA_KEY="pamap50"
DATA_NAME="PAMAP50"
RUN_ID="${GLOBAL_TIME_TAG}_pamap50_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/pamap50_50hz"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=100
SEQ_LEN=100
PATCH_LEN=50
PAMAP_VARIANT="pamap50"
TEST_SUBJECTS="subject105,subject106"
BATCH_SIZE=32
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id PAMAP2 --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id PAMAP2 --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done PAMAP50."

# ========================================================
# 5. WISDM
# ========================================================
clean_cache

echo "[5/7] Running WISDM..."
DATA_ROOT="/root/autodl-tmp/datasets/WISDM_ar_latest/WISDM_ar_v1.1"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/WISDM_ar_latest/WISDM_ar_v1.1"   # <- 改成你的路径(文件夹内有 WISDM_ar_v1.1_raw.txt) 或直接指向 raw.txt

DATA_KEY="wisdm"
DATA_NAME="WISDM"
RUN_ID="${GLOBAL_TIME_TAG}_wisdm_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/wisdm"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=80
SEQ_LEN=80
PATCH_LEN=40
BATCH_SIZE=64
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id WISDM --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  --test_users 33,34,35,36 \
  --val_users 5,13,17,19,27,31 \
  --wisdm_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done WISDM."

# ========================================================
# 6. HHAR_1user
# ========================================================
clean_cache

echo "[6/7] Running HHAR_1user..."
DATA_ROOT="/root/autodl-tmp/datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"   # <- 改成你的路径(含 Phones_*.csv / Watch_*.csv)

DATA_KEY="hhar"
DATA_NAME="HHAR_1user"
RUN_ID="${GLOBAL_TIME_TAG}_hhar_1user_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/hhar_1user"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=128
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_1user --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done HHAR_1user."

# ========================================================
# 7. MotionSense
# ========================================================
clean_cache

echo "[7/7] Running MotionSense..."
DATA_ROOT="/root/autodl-tmp/datasets/motion-sense-master/motion-sense-master/data"
#DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/motion-sense-master/motion-sense-master/data"  # <- 改成你的路径

DATA_KEY="motionsense"
DATA_NAME="MotionSense"
RUN_ID="${GLOBAL_TIME_TAG}_motionsense_ce_only"
LOG_DIR="$GLOBAL_LOG_ROOT/motionsense"
mkdir -p "$LOG_DIR"

ALIGN_W_MAX=128
SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=8

# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MotionSense --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 --diagnose_vq 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --itr 1 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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
  --llama_name "$LLAMA_NAME" --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --itr 5 \
  --use_hard_label $use_hard_label \
  --mask_mode $mask_mode \
  --mask_rate $MASK_RATE \
  --lambda_distill $LAMBDA_DISTILL \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path $VQVAEPATH \
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



python /root/autodl-tmp/SensorLLMLib_v2/send_email.py


shutdown -h now




