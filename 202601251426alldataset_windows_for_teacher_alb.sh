#!/bin/bash








use_teacher=1
distill_type="soft_kl"
teacher_init="pretrained"
teacher_use_prompt=1
lambda_distill=1.0




#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------


#set -e  # 遇到错误立即停止。如果希望忽略错误继续跑下一个，请注释掉这一行

# ===== 全局设置 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


LLAMA_NAME="D:\fuy\MyCode/Llama-3.2-1B"

TS_BACNBONE_YAML="ts_backbone.yaml"


#blation-6：Transformer
echo "Ablation-6：Transformer"


VQVAEPATH="qua_recon_path"


for i in 1;do



# 生成一个全局时间标签，这样这一次批量运行的所有日志都在同一个大目录下
GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG_ROOT="./run_log/batch_${GLOBAL_TIME_TAG}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting Batch Training Run: $GLOBAL_TIME_TAG"
echo "Logs will be saved to: $GLOBAL_LOG_ROOT"
echo "========================================================"

model_name=SensorLLMFuy_test_withllm_mae_vqvae_alb_transformer
PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"








# ========================================================
# 1. UCIHAR
# ========================================================


TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v2"

if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
    echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi






MASK_RATE=0.7

echo "[1/7] Running UCIHAR..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
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
PRETRAIN_EPOCHS=20
EPOCHS=8




# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done UCIHAR."

# ========================================================
# 2. USC-HAD
# ========================================================


# ========================================================
# 3. MHEALTH
# ========================================================








PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"

# ========================================================
# 1. WISDM
# ========================================================
TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v2"

if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
    echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi


echo "[5/7] Running WISDM..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/WISDM_ar_latest/WISDM_ar_v1.1"   # <- 改成你的路径(文件夹内有 WISDM_ar_v1.1_raw.txt) 或直接指向 raw.txt
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
#
# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id WISDM --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
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
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
  --test_users 33,34,35,36 \
  --val_users 5,13,17,19,27,31 \
  --wisdm_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done WISDM."


# ========================================================
# 2. HHAR_1user
# ========================================================

TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v2"

if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
    echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi
MASK_RATE=0.4
echo "[6/7] Running HHAR_1user..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"   # <- 改成你的路径(含 Phones_*.csv / Watch_*.csv)
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
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
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
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done HHAR_1user."




MASK_RATE=0.7

# ========================================================
# 4. MotionSense
# ========================================================

TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v2"

if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
    echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi

echo "[7/7] Running MotionSense..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/motion-sense-master/motion-sense-master/data"  # <- 改成你的路径
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
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
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
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
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


#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------


#set -e  # 遇到错误立即停止。如果希望忽略错误继续跑下一个，请注释掉这一行

# ===== 全局设置 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


LLAMA_NAME="D:\fuy\MyCode/Llama-3.2-1B"

TS_BACNBONE_YAML="ts_backbone.yaml"


#echo "Ablation-7：Linear"
echo "Ablation-7：Linear"


VQVAEPATH="qua_recon_path"


for i in 1;do



# 生成一个全局时间标签，这样这一次批量运行的所有日志都在同一个大目录下
GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG_ROOT="./run_log/batch_${GLOBAL_TIME_TAG}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting Batch Training Run: $GLOBAL_TIME_TAG"
echo "Logs will be saved to: $GLOBAL_LOG_ROOT"
echo "========================================================"

model_name=SensorLLMFuy_test_withllm_mae_vqvae_alb_linear
PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"








# ========================================================
# 1. UCIHAR
# ========================================================


TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v2"

if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
    echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi






MASK_RATE=0.7

echo "[1/7] Running UCIHAR..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
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
PRETRAIN_EPOCHS=20
EPOCHS=8




# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done UCIHAR."

# ========================================================
# 2. USC-HAD
# ========================================================


# ========================================================
# 3. MHEALTH
# ========================================================








PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"

# ========================================================
# 1. WISDM
# ========================================================
TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v2"

if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
    echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi


echo "[5/7] Running WISDM..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/WISDM_ar_latest/WISDM_ar_v1.1"   # <- 改成你的路径(文件夹内有 WISDM_ar_v1.1_raw.txt) 或直接指向 raw.txt
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
#
# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id WISDM --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
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
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
  --test_users 33,34,35,36 \
  --val_users 5,13,17,19,27,31 \
  --wisdm_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done WISDM."


# ========================================================
# 2. HHAR_1user
# ========================================================

TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v2"

if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
    echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi
MASK_RATE=0.4
echo "[6/7] Running HHAR_1user..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"   # <- 改成你的路径(含 Phones_*.csv / Watch_*.csv)
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
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
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
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done HHAR_1user."




MASK_RATE=0.7

# ========================================================
# 4. MotionSense
# ========================================================

TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v2"

if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
    echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi

echo "[7/7] Running MotionSense..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/motion-sense-master/motion-sense-master/data"  # <- 改成你的路径
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
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
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
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
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











#set -e  # 遇到错误立即停止。如果希望忽略错误继续跑下一个，请注释掉这一行

# ===== 全局设置 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


LLAMA_NAME="D:\fuy\MyCode/Llama-3.2-1B"

TS_BACNBONE_YAML="ts_backbone.yaml"

use_teacher=1
distill_type="soft_kl"
teacher_init="pretrained"
teacher_use_prompt=1
lambda_distill=1.0




#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------


#set -e  # 遇到错误立即停止。如果希望忽略错误继续跑下一个，请注释掉这一行

# ===== 全局设置 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


LLAMA_NAME="D:\fuy\MyCode/Llama-3.2-1B"

TS_BACNBONE_YAML="ts_backbone.yaml"


#Ablation-5：w/o Prompt（pretrained teacher + no prompt + soft_kl）
## ===== A5: w/o prompt =====
#teacher_init: "pretrained"
#teacher_use_prompt: false
#distill_type: "soft_kl"
echo "Ablation-5：w/o Prompt（pretrained teacher + no prompt + soft_kl）"

teacher_init="pretrained"
teacher_use_prompt=0
distill_type="soft_kl"

VQVAEPATH="qua_recon_path"


for i in 1;do



# 生成一个全局时间标签，这样这一次批量运行的所有日志都在同一个大目录下
GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG_ROOT="./run_log/batch_${GLOBAL_TIME_TAG}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting Batch Training Run: $GLOBAL_TIME_TAG"
echo "Logs will be saved to: $GLOBAL_LOG_ROOT"
echo "========================================================"

model_name=SensorLLMFuy_test_withllm_mae_vqvae_alb
PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"








# ========================================================
# 1. UCIHAR
# ========================================================


TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v2"

if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
    echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi






MASK_RATE=0.7

echo "[1/7] Running UCIHAR..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
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
PRETRAIN_EPOCHS=20
EPOCHS=8




# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage1.log" 2>&1

# Stage 2
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 2 \
  --batch_size $BATCH_SIZE --trainable_modules $TRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done UCIHAR."

# ========================================================
# 2. USC-HAD
# ========================================================


# ========================================================
# 3. MHEALTH
# ========================================================








PRETRAIN_trainable_modules="patch_embed,resampler,mae_decoder"
TRAIN_trainable_modules="patch_embed,resampler,llm_proj,cls_head"

# ========================================================
# 1. WISDM
# ========================================================
TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v2"

if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
    echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi


echo "[5/7] Running WISDM..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/WISDM_ar_latest/WISDM_ar_v1.1"   # <- 改成你的路径(文件夹内有 WISDM_ar_v1.1_raw.txt) 或直接指向 raw.txt
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
#
# Stage 1
python -u run.py \
  --task_name classification --is_training 1 --root_path "$DATA_ROOT" \
  --model_id WISDM --run_id "$RUN_ID" --datasets $DATA_NAME \
  --model "$model_name" --data $DATA_NAME --dataset_key $DATA_KEY \
  --seq_len $ALIGN_W_MAX --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE --trainable_modules $PRETRAIN_trainable_modules \
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
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
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
  --test_users 33,34,35,36 \
  --val_users 5,13,17,19,27,31 \
  --wisdm_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done WISDM."


# ========================================================
# 2. HHAR_1user
# ========================================================

TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v2"

if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
    echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi
MASK_RATE=0.4
echo "[6/7] Running HHAR_1user..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"   # <- 改成你的路径(含 Phones_*.csv / Watch_*.csv)
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
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
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
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  > "$LOG_DIR/stage2.log" 2>&1

echo "Done HHAR_1user."




MASK_RATE=0.7

# ========================================================
# 4. MotionSense
# ========================================================

TARGET_DIR="/root/autodl-tmp/SensorLLMLib_v2"

if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    find "$TARGET_DIR" -type f \( -name "*.npy" -o -name "*.pth" \) -delete
    echo "已删除 $TARGET_DIR 下所有 .npy 和 .pth 文件"
else
    echo "错误：目录不存在或变量为空，跳过删除"
fi

echo "[7/7] Running MotionSense..."
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/motion-sense-master/motion-sense-master/data"  # <- 改成你的路径
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
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
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
  --llama_name $LLAMA_NAME --use_teacher $use_teacher --distill_type $distill_type --lambda_distill $lambda_distill --teacher_use_prompt $teacher_use_prompt --teacher_init $teacher_init --learning_rate $LR --train_epochs $EPOCHS \
  --num_workers 0 --mask_rate $MASK_RATE --ts_backbone_yaml $TS_BACNBONE_YAML --vqvae_path $VQVAEPATH \
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



