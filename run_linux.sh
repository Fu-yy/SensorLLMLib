#!/bin/bash
set -e

# =========================
# 0. 自动切到脚本所在目录（关键）
# =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =========================
# 1. 时间 & 路径
# =========================
TIME_TAG=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${TIME_TAG}_mhealth_sensorllm_Ada"

LOG_ROOT="./run_log/log_${TIME_TAG}/mhealth"
mkdir -p "$LOG_ROOT"

echo "RUN_ID = $RUN_ID"
echo "LOG_ROOT = $LOG_ROOT"

# =========================
# 2. 公共配置
# =========================
DATA_ROOT="./datasets/MHEALTHDATASET/"
SEQ_LEN=200
BATCH_SIZE=16
LR=0.001
EPOCHS=8

# =========================
# 3. SensorLLMFuy
# =========================
model_name=SensorLLMFuy

# -------- Stage 1 --------
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path "$DATA_ROOT" \
  --model_id MHealth \
  --run_id "$RUN_ID" \
  --datasets MHEALTHY \
  --model "$model_name" \
  --data MHealth \
  --seq_len $SEQ_LEN \
  --e_layers 2 \
  --batch_size $BATCH_SIZE \
  --d_model 16 \
  --d_ff 32 \
  --top_k 3 \
  --des Exp \
  --itr 1 \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --patience 7 \
  --num_workers 0 \
  --dataset_key mhealth \
  --stage 1 \
  > "$LOG_ROOT/model_${model_name}_pretrain.log" 2>&1

# -------- Stage 2 --------
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path "$DATA_ROOT" \
  --model_id MHealth \
  --run_id "$RUN_ID" \
  --datasets MHEALTHY \
  --model "$model_name" \
  --data MHealth \
  --seq_len $SEQ_LEN \
  --e_layers 2 \
  --batch_size $BATCH_SIZE \
  --d_model 16 \
  --d_ff 32 \
  --top_k 3 \
  --des Exp \
  --itr 1 \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --patience 7 \
  --num_workers 0 \
  --dataset_key mhealth \
  --stage 2 \
  > "$LOG_ROOT/model_${model_name}_finetune.log" 2>&1



#  ) USC-HAD：run_uschad.sh（严格按论文）


























# =========================
# 4. TimesNet
# =========================
model_name=TimesNet

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path "$DATA_ROOT" \
  --model_id MHealth \
  --run_id "$RUN_ID" \
  --datasets MHEALTHY \
  --model "$model_name" \
  --data MHealth \
  --seq_len $SEQ_LEN \
  --e_layers 2 \
  --batch_size $BATCH_SIZE \
  --d_model 16 \
  --d_ff 32 \
  --top_k 3 \
  --des Exp \
  --itr 1 \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --patience 7 \
  --num_workers 0 \
  --dataset_key mhealth \
  --stage 2 \
  > "$LOG_ROOT/model_${model_name}_finetune.log" 2>&1
