#!/bin/bash
# 不建议 set -e，某个数据集失败时可以继续跑后面的
# set -e

# ========================================================
# 全局设置
# ========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LLAMA_NAME="/root/autodl-tmp/Llama-3.2-1B"
TS_BACKBONE_YAML="ts_backbone_linux.yaml"
LLAMA_NAME="D:\fuy\MyCode/Llama-3.2-1B"
LLAMA_NAME="D:\fuy\MyCode/Llama-3.2-1B-Instruct"
LLAMA_NAME="D:\fuy\MyCode\Qwen2.5-1.5B-Instruct"

TS_BACKBONE_YAML="ts_backbone.yaml"
# 这里对应你 yaml 里的 VQ-VAE 路径 key
# 可选：qua_path / freq_path / recon_path / qua_freq_path / qua_recon_path / freq_recon_path / all_path
VQVAEPATH="qua_recon_path"

# 新模型：你需要新建 models/PrimitivePromptLLM.py，并在 model_dict 里注册
MODEL_NAME="PrimitivePromptLLM"

# profile 生成设置
AUTO_BUILD_PROFILE=1
FORCE_BUILD_PROFILE=0

# 如果你改了 profile 代码，或者想重新生成，改成 1
# FORCE_BUILD_PROFILE=1

PRIMITIVE_PROFILE_EXAMPLES=5
PRIMITIVE_PROFILE_MIN_VALID_RATIO=0.5

# prompt 设置
PROMPT_MODE="no_label"
PROMPT_MAX_PRIMITIVES=16
PROMPT_MAX_NEW_TOKENS=128
PROMPT_SAVE_TEXT=1

# 注意：prompt-LLM 每个样本都会调用 generate，强烈建议 batch_size=1
PROMPT_BATCH_SIZE=1

GLOBAL_TIME_TAG=$(date +"%Y%m%d_%H%M%S")
GLOBAL_LOG_ROOT="./run_log/prompt_llm_eval_${GLOBAL_TIME_TAG}_${PROMPT_MODE}"
mkdir -p "$GLOBAL_LOG_ROOT"

echo "========================================================"
echo "Starting Prompt-LLM Zero-shot Evaluation: $GLOBAL_TIME_TAG"
echo "Logs will be saved to: $GLOBAL_LOG_ROOT"
echo "LLM: $LLAMA_NAME"
echo "VQ path key: $VQVAEPATH"
echo "Prompt mode: $PROMPT_MODE"
echo "========================================================"


# ========================================================
# 通用运行函数
# ========================================================

run_prompt_eval () {
  DATA_ROOT="$1"
  DATA_KEY="$2"
  DATA_NAME="$3"
  MODEL_ID="$4"
  SEQ_LEN="$5"
  PATCH_LEN="$6"
  BATCH_SIZE="$7"
  LOG_SUBDIR="$8"
  EXTRA_ARGS="$9"

  RUN_ID="${GLOBAL_TIME_TAG}_${DATA_KEY}_prompt_llm"
  LOG_DIR="$GLOBAL_LOG_ROOT/$LOG_SUBDIR"
  mkdir -p "$LOG_DIR"

  echo "========================================================"
  echo "Running Prompt-LLM Eval: $DATA_NAME"
  echo "DATA_KEY=$DATA_KEY"
  echo "SEQ_LEN=$SEQ_LEN PATCH_LEN=$PATCH_LEN"
  echo "LOG_DIR=$LOG_DIR"
  echo "========================================================"

  python -u run.py \
    --task_name alignment \
    --is_training 0 \
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
    --batch_size "$BATCH_SIZE" \
    --num_workers 0 \
    --llama_name "$LLAMA_NAME" \
    --ts_backbone_yaml "$TS_BACKBONE_YAML" \
    --vqvae_path "$VQVAEPATH" \
    --auto_build_primitive_profile "$AUTO_BUILD_PROFILE" \
    --force_build_primitive_profile "$FORCE_BUILD_PROFILE" \
    --primitive_profile_examples "$PRIMITIVE_PROFILE_EXAMPLES" \
    --primitive_profile_min_valid_ratio "$PRIMITIVE_PROFILE_MIN_VALID_RATIO" \
    --prompt_mode "$PROMPT_MODE" \
    --prompt_max_primitives "$PROMPT_MAX_PRIMITIVES" \
    --prompt_max_new_tokens "$PROMPT_MAX_NEW_TOKENS" \
    --prompt_save_text "$PROMPT_SAVE_TEXT" \
    $EXTRA_ARGS \
    > "$LOG_DIR/prompt_eval.log" 2>&1

  echo "Done Prompt-LLM Eval: $DATA_NAME"
}


# ========================================================
# 1. UCIHAR
# ========================================================

DATA_ROOT="/root/autodl-tmp/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"

DATA_KEY="ucihar"
DATA_NAME="UCIHAR"
MODEL_ID="UCIHAR_PromptLLM"
SEQ_LEN=128
PATCH_LEN=64

run_prompt_eval \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "$PROMPT_BATCH_SIZE" \
  "ucihar" \
  ""


# ========================================================
# 2. USC-HAD
# ========================================================

DATA_ROOT="/root/autodl-tmp/datasets/USC-HAD/USC-HAD"
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/USC-HAD/USC-HAD"

DATA_KEY="uschad"
DATA_NAME="USCHAD"
MODEL_ID="USCHAD_PromptLLM"
SEQ_LEN=200
PATCH_LEN=100
TEST_SUBJECTS="subject13,subject14"

run_prompt_eval \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "$PROMPT_BATCH_SIZE" \
  "uschad" \
  "--test_subjects $TEST_SUBJECTS"


# ========================================================
# 3. MHEALTH
# ========================================================

DATA_ROOT="/root/autodl-tmp/datasets/MHEALTHDATASET"
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/MHEALTHDATASET"

DATA_KEY="mhealth"
DATA_NAME="MHealth"
MODEL_ID="MHealth_PromptLLM"
SEQ_LEN=100
PATCH_LEN=50
TEST_SUBJECTS="subject1,subject3,subject6"

run_prompt_eval \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "$PROMPT_BATCH_SIZE" \
  "mhealth" \
  "--test_subjects $TEST_SUBJECTS"


# ========================================================
# 4. PAMAP2 50Hz
# ========================================================

DATA_ROOT="/root/autodl-tmp/datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"

DATA_KEY="pamap50"
DATA_NAME="PAMAP50"
MODEL_ID="PAMAP50_PromptLLM"
SEQ_LEN=100
PATCH_LEN=50
PAMAP_VARIANT="pamap50"
TEST_SUBJECTS="subject105,subject106"

run_prompt_eval \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "$PROMPT_BATCH_SIZE" \
  "pamap50" \
  "--pamap_variant $PAMAP_VARIANT --test_subjects $TEST_SUBJECTS"


# ========================================================
# 5. WISDM
# ========================================================

DATA_ROOT="/root/autodl-tmp/datasets/WISDM_ar_latest/WISDM_ar_v1.1"
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/WISDM_ar_latest/WISDM_ar_v1.1"   # <- 改成你的路径(文件夹内有 WISDM_ar_v1.1_raw.txt) 或直接指向 raw.txt

DATA_KEY="wisdm"
DATA_NAME="WISDM"
MODEL_ID="WISDM_PromptLLM"
SEQ_LEN=80
PATCH_LEN=40

run_prompt_eval \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "$PROMPT_BATCH_SIZE" \
  "wisdm" \
  "--test_users 33,34,35,36 --val_users 5,13,17,19,27,31 --wisdm_norm none"


# ========================================================
# 6. HHAR_1user
# ========================================================

DATA_ROOT="/root/autodl-tmp/datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp"   # <- 改成你的路径(含 Phones_*.csv / Watch_*.csv)

DATA_KEY="hhar"
DATA_NAME="HHAR_1user"
MODEL_ID="HHAR_1user_PromptLLM"
SEQ_LEN=128
PATCH_LEN=64

run_prompt_eval \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "$PROMPT_BATCH_SIZE" \
  "hhar_1user" \
  "--hhar_tol 0.05 --hhar_align_on Arrival_Time --hhar_use_cache 1 --hhar_norm none"


# ========================================================
# 7. MotionSense
# ========================================================

DATA_ROOT="/root/autodl-tmp/datasets/motion-sense-master/motion-sense-master/data"
DATA_ROOT="D:\fuy\MyCode\SensorLLMLib\datasets/motion-sense-master/motion-sense-master/data"  # <- 改成你的路径

DATA_KEY="motionsense"
DATA_NAME="MotionSense"
MODEL_ID="MotionSense_PromptLLM"
SEQ_LEN=128
PATCH_LEN=64

run_prompt_eval \
  "$DATA_ROOT" \
  "$DATA_KEY" \
  "$DATA_NAME" \
  "$MODEL_ID" \
  "$SEQ_LEN" \
  "$PATCH_LEN" \
  "$PROMPT_BATCH_SIZE" \
  "motionsense" \
  "--test_users 19,20,21,22,23,24 --val_users 13,14,15,16,17,18 --motionsense_feature_set A12 --motionsense_combine_grav_acc 0 --motionsense_norm none"


echo "========================================================"
echo "ALL PROMPT-LLM EVAL DONE."
echo "Logs at: $GLOBAL_LOG_ROOT"
echo "========================================================"