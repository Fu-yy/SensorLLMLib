#!/bin/bash
#set -e

# ========================================================
# Teacher Alignment Script
# Masked Primitive Prediction Alignment
# Train: projector + mask_embed_llama + output_head
# Freeze: VQ-VAE + LLM
# ========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ===== Global Settings =====
LLAMA_NAME="D:/fuy/MyCode/Llama-3.2-1B"
LLAMA_NAME="/root/autodl-tmp/Llama-3.2-1B"

TS_BACKBONE_YAML="ts_backbone.yaml"
TS_BACKBONE_YAML="ts_backbone_linux.yaml"

VQVAEPATH="qua_recon_path"

MODEL_NAME="Alignment_Stage"

# New alignment setting
ALIGN_MASK_RATE=0.4
MASK_RATE=0.4

PATIENCE=10
PRETRAIN_EPOCHS=40
LR=0.001

# Only these modules should be trained during alignment
PRETRAIN_trainable_modules="projector,output_head,mask_embed_llama,primitive_pos_embed,query_embed"
GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_TIME_TAG="Alignment_mask${ALIGN_MASK_RATE}"
GLOBAL_LOG_ROOT="./run_log/batch_${GLOBAL_TIME_TAG}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting Teacher Alignment Run: $GLOBAL_TIME_TAG"
echo "Alignment objective: masked primitive prediction"
echo "Trainable modules: $PRETRAIN_trainable_modules"
echo "LLM: $LLAMA_NAME"
echo "VQ path key: $VQVAEPATH"
echo "align_mask_rate: $ALIGN_MASK_RATE"
echo "Logs will be saved to: $GLOBAL_LOG_ROOT"
echo "========================================================"

for i in 1; do

# ========================================================
# 1. UCIHAR
# ========================================================
echo "[1/7] Running UCIHAR Alignment..."

DATA_ROOT="/root/autodl-tmp/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
DATA_KEY="ucihar"
DATA_NAME="UCIHAR"
RUN_ID="${GLOBAL_TIME_TAG}_ucihar_alignment"
LOG_DIR="$GLOBAL_LOG_ROOT/ucihar"
mkdir -p "$LOG_DIR"

SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=32

python -u run.py \
  --task_name alignment --is_training 1 --root_path "$DATA_ROOT" \
  --model_id UCIHAR --run_id "$RUN_ID" --datasets "$DATA_NAME" \
  --model "$MODEL_NAME" --data "$DATA_NAME" --dataset_key "$DATA_KEY" \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE \
  --trainable_modules "$PRETRAIN_trainable_modules" \
  --llama_name "$LLAMA_NAME" \
  --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS --patience $PATIENCE \
  --num_workers 0 \
  --align_mask_rate $ALIGN_MASK_RATE \
  --mask_rate $MASK_RATE \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path "$VQVAEPATH" \
  > "$LOG_DIR/stage1.log" 2>&1

echo "Done UCIHAR Alignment."

# ========================================================
# 2. USC-HAD
# ========================================================
echo "[2/7] Running USC-HAD Alignment..."

DATA_ROOT="/root/autodl-tmp/datasets/USC-HAD/USC-HAD"
DATA_KEY="uschad"
DATA_NAME="USCHAD"
RUN_ID="${GLOBAL_TIME_TAG}_uschad_alignment"
LOG_DIR="$GLOBAL_LOG_ROOT/uschad"
mkdir -p "$LOG_DIR"

SEQ_LEN=200
PATCH_LEN=100
BATCH_SIZE=16
TEST_SUBJECTS="subject13,subject14"

python -u run.py \
  --task_name alignment --is_training 1 --root_path "$DATA_ROOT" \
  --model_id USCHAD --run_id "$RUN_ID" --datasets "$DATA_NAME" \
  --model "$MODEL_NAME" --data "$DATA_NAME" --dataset_key "$DATA_KEY" \
  --test_subjects "$TEST_SUBJECTS" \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE \
  --trainable_modules "$PRETRAIN_trainable_modules" \
  --llama_name "$LLAMA_NAME" \
  --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS --patience $PATIENCE \
  --num_workers 0 \
  --align_mask_rate $ALIGN_MASK_RATE \
  --mask_rate $MASK_RATE \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path "$VQVAEPATH" \
  > "$LOG_DIR/stage1.log" 2>&1

echo "Done USC-HAD Alignment."

# ========================================================
# 3. MHEALTH
# ========================================================
echo "[3/7] Running MHEALTH Alignment..."

DATA_ROOT="/root/autodl-tmp/datasets/MHEALTHDATASET"
DATA_KEY="mhealth"
DATA_NAME="MHealth"
RUN_ID="${GLOBAL_TIME_TAG}_mhealth_alignment"
LOG_DIR="$GLOBAL_LOG_ROOT/mhealth"
mkdir -p "$LOG_DIR"

SEQ_LEN=100
PATCH_LEN=50
BATCH_SIZE=16
TEST_SUBJECTS="subject1,subject3,subject6"

python -u run.py \
  --task_name alignment --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MHealth --run_id "$RUN_ID" --datasets "$DATA_NAME" \
  --model "$MODEL_NAME" --data "$DATA_NAME" --dataset_key "$DATA_KEY" \
  --test_subjects "$TEST_SUBJECTS" \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE \
  --trainable_modules "$PRETRAIN_trainable_modules" \
  --llama_name "$LLAMA_NAME" \
  --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS --patience $PATIENCE \
  --num_workers 0 \
  --align_mask_rate $ALIGN_MASK_RATE \
  --mask_rate $MASK_RATE \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path "$VQVAEPATH" \
  > "$LOG_DIR/stage1.log" 2>&1

echo "Done MHEALTH Alignment."

# ========================================================
# 4. PAMAP2 50Hz
# ========================================================
echo "[4/7] Running PAMAP2 50Hz Alignment..."

DATA_ROOT="/root/autodl-tmp/datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"
DATA_KEY="pamap50"
DATA_NAME="PAMAP50"
RUN_ID="${GLOBAL_TIME_TAG}_pamap50_alignment"
LOG_DIR="$GLOBAL_LOG_ROOT/pamap50_50hz"
mkdir -p "$LOG_DIR"

SEQ_LEN=100
PATCH_LEN=50
BATCH_SIZE=32
PAMAP_VARIANT="pamap50"
TEST_SUBJECTS="subject105,subject106"

python -u run.py \
  --task_name alignment --is_training 1 --root_path "$DATA_ROOT" \
  --model_id PAMAP2 --run_id "$RUN_ID" --datasets "$DATA_NAME" \
  --model "$MODEL_NAME" --data "$DATA_NAME" --dataset_key "$DATA_KEY" \
  --pamap_variant "$PAMAP_VARIANT" --test_subjects "$TEST_SUBJECTS" \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE \
  --trainable_modules "$PRETRAIN_trainable_modules" \
  --llama_name "$LLAMA_NAME" \
  --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS --patience $PATIENCE \
  --num_workers 0 \
  --align_mask_rate $ALIGN_MASK_RATE \
  --mask_rate $MASK_RATE \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path "$VQVAEPATH" \
  > "$LOG_DIR/stage1.log" 2>&1

echo "Done PAMAP2 50Hz Alignment."

# ========================================================
# 5. WISDM
# ========================================================
echo "[5/7] Running WISDM Alignment..."

DATA_ROOT="/root/autodl-tmp/datasets/WISDM_ar_latest/WISDM_ar_v1.1"
DATA_KEY="wisdm"
DATA_NAME="WISDM"
RUN_ID="${GLOBAL_TIME_TAG}_wisdm_alignment"
LOG_DIR="$GLOBAL_LOG_ROOT/wisdm"
mkdir -p "$LOG_DIR"

SEQ_LEN=80
PATCH_LEN=40
BATCH_SIZE=64

python -u run.py \
  --task_name alignment --is_training 1 --root_path "$DATA_ROOT" \
  --model_id WISDM --run_id "$RUN_ID" --datasets "$DATA_NAME" \
  --model "$MODEL_NAME" --data "$DATA_NAME" --dataset_key "$DATA_KEY" \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE \
  --trainable_modules "$PRETRAIN_trainable_modules" \
  --llama_name "$LLAMA_NAME" \
  --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS --patience $PATIENCE \
  --num_workers 0 \
  --align_mask_rate $ALIGN_MASK_RATE \
  --mask_rate $MASK_RATE \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path "$VQVAEPATH" \
  --test_users 33,34,35,36 \
  --val_users 5,13,17,19,27,31 \
  --wisdm_norm none \
  > "$LOG_DIR/stage1.log" 2>&1

echo "Done WISDM Alignment."

# ========================================================
# 6. HHAR_1user
# ========================================================
echo "[6/7] Running HHAR_1user Alignment..."

DATA_ROOT="/root/autodl-tmp/datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"
DATA_KEY="hhar"
DATA_NAME="HHAR_1user"
RUN_ID="${GLOBAL_TIME_TAG}_hhar_1user_alignment"
LOG_DIR="$GLOBAL_LOG_ROOT/hhar_1user"
mkdir -p "$LOG_DIR"

SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64

python -u run.py \
  --task_name alignment --is_training 1 --root_path "$DATA_ROOT" \
  --model_id HHAR_1user --run_id "$RUN_ID" --datasets "$DATA_NAME" \
  --model "$MODEL_NAME" --data "$DATA_NAME" --dataset_key "$DATA_KEY" \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE \
  --trainable_modules "$PRETRAIN_trainable_modules" \
  --llama_name "$LLAMA_NAME" \
  --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS --patience $PATIENCE \
  --num_workers 0 \
  --align_mask_rate $ALIGN_MASK_RATE \
  --mask_rate $MASK_RATE \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path "$VQVAEPATH" \
  --hhar_tol 0.05 \
  --hhar_align_on Arrival_Time \
  --hhar_use_cache 1 \
  --hhar_norm none \
  > "$LOG_DIR/stage1.log" 2>&1

echo "Done HHAR_1user Alignment."

# ========================================================
# 7. MotionSense
# ========================================================
echo "[7/7] Running MotionSense Alignment..."

DATA_ROOT="/root/autodl-tmp/datasets/motion-sense-master/motion-sense-master/data"
DATA_KEY="motionsense"
DATA_NAME="MotionSense"
RUN_ID="${GLOBAL_TIME_TAG}_motionsense_alignment"
LOG_DIR="$GLOBAL_LOG_ROOT/motionsense"
mkdir -p "$LOG_DIR"

SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64

python -u run.py \
  --task_name alignment --is_training 1 --root_path "$DATA_ROOT" \
  --model_id MotionSense --run_id "$RUN_ID" --datasets "$DATA_NAME" \
  --model "$MODEL_NAME" --data "$DATA_NAME" --dataset_key "$DATA_KEY" \
  --seq_len $SEQ_LEN --patch_len $PATCH_LEN --stride $PATCH_LEN --stage 1 \
  --batch_size $BATCH_SIZE \
  --trainable_modules "$PRETRAIN_trainable_modules" \
  --llama_name "$LLAMA_NAME" \
  --learning_rate $LR --train_epochs $PRETRAIN_EPOCHS --patience $PATIENCE \
  --num_workers 0 \
  --align_mask_rate $ALIGN_MASK_RATE \
  --mask_rate $MASK_RATE \
  --ts_backbone_yaml $TS_BACKBONE_YAML \
  --vqvae_path "$VQVAEPATH" \
  --test_users 19,20,21,22,23,24 \
  --val_users 13,14,15,16,17,18 \
  --motionsense_feature_set A12 \
  --motionsense_combine_grav_acc 0 \
  --motionsense_norm none \
  > "$LOG_DIR/stage1.log" 2>&1

echo "Done MotionSense Alignment."

echo "========================================================"
echo "ALL DONE. Logs at: $GLOBAL_LOG_ROOT"
echo "========================================================"

done

python /root/autodl-tmp/SensorLLMLib_v2/send_email.py


shutdown -h now
