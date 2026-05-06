#!/bin/bash
#set -e

# ========================================================
# Primitive-Language Teacher Alignment Script
#
# Objective:
#   1. Masked primitive recovery
#   2. Activity classification
#
# Train:
#   teacher_core.projector
#   teacher_core.output_head
#   teacher_core.activity_head
#   teacher_core.mask_embed_llama
#   teacher_core.primitive_pos_embed
#   teacher_core.query_embed
#
# Freeze:
#   VQ-VAE
#   LLM backbone
#
# Required code support:
#   --lambda_activity
#   --use_semantic_primitive
#   --semantic_weight
#   --auto_build_primitive_profile
#   --pretrain_monitor
# ========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ========================================================
# Global Settings
# ========================================================

# Local / Linux path. The second assignment overrides the first.
LLAMA_NAME="D:/fuy/MyCode/Llama-3.2-1B"
LLAMA_NAME="/root/autodl-tmp/Llama-3.2-1B"

TS_BACKBONE_YAML="ts_backbone.yaml"
TS_BACKBONE_YAML="ts_backbone_linux.yaml"

# VQ-VAE checkpoint key in YAML.
# Options usually include:
#   qua_path
#   freq_path
#   recon_path
#   qua_freq_path
#   qua_recon_path
#   freq_recon_path
#   all_path
VQVAEPATH="qua_recon_path"

MODEL_NAME="Alignment_Stage"

# ========================================================
# New primitive-language teacher settings
# ========================================================

ALIGN_MASK_RATE=0.4
MASK_RATE=0.4

# Multi-task loss:
#   loss = primitive_loss + LAMBDA_ACTIVITY * activity_loss
LAMBDA_ACTIVITY=0.5

# Semantic primitive embedding from primitive_profile.json
USE_SEMANTIC_PRIMITIVE=1
SEMANTIC_WEIGHT=0.5

# If primitive_profile.json does not exist under the VQ-VAE ckpt dir,
# Exp will build it before initializing AlignmentModel.
AUTO_BUILD_PRIMITIVE_PROFILE=1

# Save best stage1 adapter according to:
#   loss / activity_acc / primitive_acc
PRETRAIN_MONITOR="activity_acc"

TEACHER_TEMPERATURE=1.0
TEACHER_PROMPT="Recover masked wearable sensor primitives and infer the human activity:"
TEACHER_PROMPT="Analyze wearable IMU motion primitives. Use primitive semantic profiles to recover masked primitives and classify the activity."
PATIENCE=10
PRETRAIN_EPOCHS=40
LR=0.001
WEIGHT_DECAY=0.0001
MIN_LR=0.00001
WARMUP_EPOCHS=0

FINETUNE_EPOCHS=8

# Stage2 classification fine-tuning.
# 可选：
#   activity_only       只训练 activity_head
#   activity_projector  训练 projector + activity_head
#   adapter_all         训练 projector/output_head/activity_head/mask/pos/query
STAGE2_TRAINABLE="activity_only"
# Stage1 trainable modules.
# If your new set_trainable_modules() uses teacher_core.freeze_llm_only(),
# this argument is not strictly necessary, but keeping it is harmless.
PRETRAIN_trainable_modules="teacher_core.projector,teacher_core.output_head,teacher_core.activity_head,teacher_core.mask_embed_llama,teacher_core.primitive_pos_embed,teacher_core.query_embed"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
METHOD_TAG="PrimitiveTeacher_mask${ALIGN_MASK_RATE}_lambda${LAMBDA_ACTIVITY}"

GLOBAL_TIME_TAG="${TIME_TAG}_${METHOD_TAG}"
GLOBAL_LOG_ROOT="./run_log/batch_${GLOBAL_TIME_TAG}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting Primitive-Language Teacher Alignment Run"
echo "Run tag: $GLOBAL_TIME_TAG"
echo "Objective: masked primitive recovery + activity classification"
echo "LLM: $LLAMA_NAME"
echo "YAML: $TS_BACKBONE_YAML"
echo "VQ path key: $VQVAEPATH"
echo "align_mask_rate: $ALIGN_MASK_RATE"
echo "lambda_activity: $LAMBDA_ACTIVITY"
echo "use_semantic_primitive: $USE_SEMANTIC_PRIMITIVE"
echo "semantic_weight: $SEMANTIC_WEIGHT"
echo "auto_build_primitive_profile: $AUTO_BUILD_PRIMITIVE_PROFILE"
echo "pretrain_monitor: $PRETRAIN_MONITOR"
echo "Logs will be saved to: $GLOBAL_LOG_ROOT"
echo "========================================================"


# ========================================================
# Common runner
# ========================================================
run_alignment () {
  DATA_ROOT="$1"
  DATA_KEY="$2"
  DATA_NAME="$3"
  MODEL_ID="$4"
  RUN_ID="$5"
  LOG_DIR="$6"
  SEQ_LEN="$7"
  PATCH_LEN="$8"
  BATCH_SIZE="$9"
  EXTRA_ARGS="${10}"
#  RUN_ID="20260506_113101"
  mkdir -p "$LOG_DIR"

  echo "--------------------------------------------------------"
  echo "Running $DATA_NAME Alignment + Classification"
  echo "DATA_ROOT: $DATA_ROOT"
  echo "DATA_KEY: $DATA_KEY"
  echo "MODEL_ID: $MODEL_ID"
  echo "RUN_ID: $RUN_ID"
  echo "SEQ_LEN: $SEQ_LEN"
  echo "PATCH_LEN: $PATCH_LEN"
  echo "BATCH_SIZE: $BATCH_SIZE"
  echo "LOG_DIR: $LOG_DIR"
  echo "EXTRA_ARGS: $EXTRA_ARGS"
  echo "--------------------------------------------------------"

  # ======================================================
  # Stage 1: primitive-language teacher alignment
  # ======================================================
  python -u run.py \
    --task_name alignment \
    --is_training 1 \
    --root_path "$DATA_ROOT" \
    --model_id "$MODEL_ID" \
    --run_id "$RUN_ID" \
    --datasets "$DATA_NAME" \
    --model "$MODEL_NAME" \
    --data "$DATA_NAME" \
    --dataset_key "$DATA_KEY" \
    --seq_len "$SEQ_LEN" \
    --patch_len "$PATCH_LEN" \
    --stride "$PATCH_LEN" \
    --stage 1 \
    --two_stage 1 \
    --batch_size "$BATCH_SIZE" \
    --trainable_modules "$PRETRAIN_trainable_modules" \
    --llama_name "$LLAMA_NAME" \
    --learning_rate "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --min_lr "$MIN_LR" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --train_epochs "$PRETRAIN_EPOCHS" \
    --patience "$PATIENCE" \
    --num_workers 0 \
    --align_mask_rate "$ALIGN_MASK_RATE" \
    --mask_rate "$MASK_RATE" \
    --lambda_activity "$LAMBDA_ACTIVITY" \
    --teacher_temperature "$TEACHER_TEMPERATURE" \
    --teacher_prompt "$TEACHER_PROMPT" \
    --use_semantic_primitive "$USE_SEMANTIC_PRIMITIVE" \
    --semantic_weight "$SEMANTIC_WEIGHT" \
    --auto_build_primitive_profile "$AUTO_BUILD_PRIMITIVE_PROFILE" \
    --force_build_primitive_profile 1 \
    --primitive_profile_examples 5 \
    --primitive_profile_min_valid_ratio 0.5 \
    --pretrain_monitor "$PRETRAIN_MONITOR" \
    --ts_backbone_yaml "$TS_BACKBONE_YAML" \
    --vqvae_path "$VQVAEPATH" \
    $EXTRA_ARGS \
    > "$LOG_DIR/stage1.log" 2>&1

  echo "Done $DATA_NAME Stage 1 Alignment."

  # ======================================================
  # Stage 2: clean activity classification fine-tuning
  # ======================================================
  python -u run.py \
    --task_name alignment \
    --is_training 1 \
    --root_path "$DATA_ROOT" \
    --model_id "$MODEL_ID" \
    --run_id "$RUN_ID" \
    --datasets "$DATA_NAME" \
    --model "$MODEL_NAME" \
    --data "$DATA_NAME" \
    --dataset_key "$DATA_KEY" \
    --seq_len "$SEQ_LEN" \
    --patch_len "$PATCH_LEN" \
    --stride "$PATCH_LEN" \
    --stage 2 \
    --two_stage 1 \
    --batch_size "$BATCH_SIZE" \
    --llama_name "$LLAMA_NAME" \
    --learning_rate "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --min_lr "$MIN_LR" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --train_epochs "$FINETUNE_EPOCHS" \
    --patience "$PATIENCE" \
    --num_workers 0 \
    --stage2_trainable "$STAGE2_TRAINABLE" \
    --align_mask_rate "$ALIGN_MASK_RATE" \
    --mask_rate "$MASK_RATE" \
    --lambda_activity "$LAMBDA_ACTIVITY" \
    --teacher_temperature "$TEACHER_TEMPERATURE" \
    --teacher_prompt "$TEACHER_PROMPT" \
    --use_semantic_primitive "$USE_SEMANTIC_PRIMITIVE" \
    --semantic_weight "$SEMANTIC_WEIGHT" \
    --auto_build_primitive_profile 0 \
    --force_build_primitive_profile 0 \
    --pretrain_monitor "$PRETRAIN_MONITOR" \
    --ts_backbone_yaml "$TS_BACKBONE_YAML" \
    --vqvae_path "$VQVAEPATH" \
    $EXTRA_ARGS \
    > "$LOG_DIR/stage2.log" 2>&1

  echo "Done $DATA_NAME Stage 2 Classification."
}

for i in 1; do

# ========================================================
# 1. UCIHAR
# ========================================================
echo "[1/7] Running UCIHAR Primitive Teacher Alignment..."

DATA_ROOT="/root/autodl-tmp/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
DATA_KEY="ucihar"
DATA_NAME="UCIHAR"
MODEL_ID="UCIHAR"
RUN_ID="${GLOBAL_TIME_TAG}_ucihar_alignment"
LOG_DIR="${GLOBAL_LOG_ROOT}/ucihar"

SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=32
EXTRA_ARGS=""

run_alignment \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$RUN_ID" \
  "$LOG_DIR" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "$BATCH_SIZE" \
  "$EXTRA_ARGS"


# ========================================================
# 2. USC-HAD
# ========================================================
echo "[2/7] Running USC-HAD Primitive Teacher Alignment..."

DATA_ROOT="/root/autodl-tmp/datasets/USC-HAD/USC-HAD"
DATA_KEY="uschad"
DATA_NAME="USCHAD"
MODEL_ID="USCHAD"
RUN_ID="${GLOBAL_TIME_TAG}_uschad_alignment"
LOG_DIR="${GLOBAL_LOG_ROOT}/uschad"

SEQ_LEN=200
PATCH_LEN=100
BATCH_SIZE=16
TEST_SUBJECTS="subject13,subject14"
EXTRA_ARGS="--test_subjects $TEST_SUBJECTS"

run_alignment \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$RUN_ID" \
  "$LOG_DIR" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "$BATCH_SIZE" \
  "$EXTRA_ARGS"


# ========================================================
# 3. MHEALTH
# ========================================================
echo "[3/7] Running MHEALTH Primitive Teacher Alignment..."

DATA_ROOT="/root/autodl-tmp/datasets/MHEALTHDATASET"
DATA_KEY="mhealth"
DATA_NAME="MHealth"
MODEL_ID="MHealth"
RUN_ID="${GLOBAL_TIME_TAG}_mhealth_alignment"
LOG_DIR="${GLOBAL_LOG_ROOT}/mhealth"

SEQ_LEN=100
PATCH_LEN=50
BATCH_SIZE=16
TEST_SUBJECTS="subject1,subject3,subject6"
EXTRA_ARGS="--test_subjects $TEST_SUBJECTS"

run_alignment \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$RUN_ID" \
  "$LOG_DIR" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "$BATCH_SIZE" \
  "$EXTRA_ARGS"


# ========================================================
# 4. PAMAP2 50Hz
# ========================================================
echo "[4/7] Running PAMAP2 50Hz Primitive Teacher Alignment..."

DATA_ROOT="/root/autodl-tmp/datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"
DATA_KEY="pamap50"
DATA_NAME="PAMAP50"
MODEL_ID="PAMAP2"
RUN_ID="${GLOBAL_TIME_TAG}_pamap50_alignment"
LOG_DIR="${GLOBAL_LOG_ROOT}/pamap50_50hz"

SEQ_LEN=100
PATCH_LEN=50
BATCH_SIZE=32
PAMAP_VARIANT="pamap50"
TEST_SUBJECTS="subject105,subject106"
EXTRA_ARGS="--pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS"

run_alignment \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$RUN_ID" \
  "$LOG_DIR" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "$BATCH_SIZE" \
  "$EXTRA_ARGS"


# ========================================================
# 5. WISDM
# ========================================================
echo "[5/7] Running WISDM Primitive Teacher Alignment..."

DATA_ROOT="/root/autodl-tmp/datasets/WISDM_ar_latest/WISDM_ar_v1.1"
DATA_KEY="wisdm"
DATA_NAME="WISDM"
MODEL_ID="WISDM"
RUN_ID="${GLOBAL_TIME_TAG}_wisdm_alignment"
LOG_DIR="${GLOBAL_LOG_ROOT}/wisdm"

SEQ_LEN=80
PATCH_LEN=40
BATCH_SIZE=64
EXTRA_ARGS="--test_users 33,34,35,36 --val_users 5,13,17,19,27,31 --wisdm_norm none"

run_alignment \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$RUN_ID" \
  "$LOG_DIR" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "$BATCH_SIZE" \
  "$EXTRA_ARGS"


# ========================================================
# 6. HHAR_1user
# ========================================================
echo "[6/7] Running HHAR_1user Primitive Teacher Alignment..."

DATA_ROOT="/root/autodl-tmp/datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"

# Important:
# Use hhar_1user here, not hhar.
# In your YAML, hhar does not contain qua_path / qua_recon_path,
# but hhar_1user does.
DATA_KEY="hhar_1user"

DATA_NAME="HHAR_1user"
MODEL_ID="HHAR_1user"
RUN_ID="${GLOBAL_TIME_TAG}_hhar_1user_alignment"
LOG_DIR="${GLOBAL_LOG_ROOT}/hhar_1user"

SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
EXTRA_ARGS="--hhar_tol 0.05 --hhar_align_on Arrival_Time --hhar_use_cache 1 --hhar_norm none"

run_alignment \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$RUN_ID" \
  "$LOG_DIR" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "$BATCH_SIZE" \
  "$EXTRA_ARGS"


# ========================================================
# 7. MotionSense
# ========================================================
echo "[7/7] Running MotionSense Primitive Teacher Alignment..."

DATA_ROOT="/root/autodl-tmp/datasets/motion-sense-master/motion-sense-master/data"
DATA_KEY="motionsense"
DATA_NAME="MotionSense"
MODEL_ID="MotionSense"
RUN_ID="${GLOBAL_TIME_TAG}_motionsense_alignment"
LOG_DIR="${GLOBAL_LOG_ROOT}/motionsense"

SEQ_LEN=128
PATCH_LEN=64
BATCH_SIZE=64
EXTRA_ARGS="--test_users 19,20,21,22,23,24 --val_users 13,14,15,16,17,18 --motionsense_feature_set A12 --motionsense_combine_grav_acc 0 --motionsense_norm none"

run_alignment \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$RUN_ID" \
  "$LOG_DIR" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "$BATCH_SIZE" \
  "$EXTRA_ARGS"


echo "========================================================"
echo "ALL DONE. Logs at: $GLOBAL_LOG_ROOT"
echo "========================================================"

done


# ========================================================
# Optional notification and shutdown
# ========================================================

python /root/autodl-tmp/SensorLLMLib_v2/send_email.py

shutdown -h now