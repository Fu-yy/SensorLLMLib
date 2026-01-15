#!/bin/bash


if [ ! -d "./run_log" ]; then
    mkdir ./run_log
fi
if [ ! -d "./run_log/log_202512192033" ]; then
    mkdir ./run_log/log_202512192033
fi
if [ ! -d "./run_log/log_202512192033/mhealth" ]; then
    mkdir ./run_log/log_202512192033/mhealth
fi


RUN_ID="$(date +"%Y%m%d_%H%M%S")_mhealth_sensorllm_Ada"
echo "RUN_ID = $RUN_ID"
model_name=SensorLLMFuy

for pred_len in 96; do
  echo "mhealth $pred_len"
   python -u  run.py \
      --task_name classification \
      --is_training 1 \
      --root_path ./datasets/MHEALTHDATASET/ \
      --model_id MHealth \
      --run_id $RUN_ID \
      --datasets MHEALTHY \
      --model $model_name \
      --data MHealth \
      --seq_len 100 \
      --e_layers 2 \
      --batch_size 16 \
      --d_model 16 \
      --d_ff 32 \
      --top_k 3 \
      --des 'Exp' \
      --itr 1 \
      --learning_rate 0.001 \
      --train_epochs 8 \
      --patience 7 \
      --num_workers 0 \
      --dataset_key mhealth \
      --stage 1 \
        > ./run_log/log_202512192033/mhealth/'model='$model_name'_pretrain_'0.01.log 2>&1
  done

for pred_len in 96; do
  echo "mhealth $pred_len"
   python -u  run.py \
      --task_name classification \
      --is_training 1 \
      --root_path ./datasets/MHEALTHDATASET/ \
      --model_id MHealth \
      --run_id $RUN_ID \
      --datasets MHEALTHY \
      --model $model_name \
      --data MHealth \
      --e_layers 2 \
      --seq_len 100 \
      --batch_size 16 \
      --d_model 16 \
      --d_ff 32 \
      --top_k 3 \
      --des 'Exp' \
      --itr 1 \
      --learning_rate 0.001 \
      --patience 7 \
      --num_workers 0 \
      --dataset_key mhealth \
      --stage 2 \
      --train_epochs 8 \
        > ./run_log/log_202512192033/mhealth/'model='$model_name'_finetune_'0.01.log 2>&1
  done


model_name=TimesNet
for pred_len in 96; do
  echo "mhealth $pred_len"
   python -u  run.py \
      --task_name classification \
      --is_training 1 \
      --root_path ./datasets/MHEALTHDATASET/ \
      --model_id MHealth \
      --run_id $RUN_ID \
      --datasets MHEALTHY \
      --model $model_name \
      --data MHealth \
      --e_layers 2 \
      --batch_size 16 \
      --seq_len 100 \
      --d_model 16 \
      --d_ff 32 \
      --top_k 3 \
      --des 'Exp' \
      --itr 1 \
      --learning_rate 0.001 \
      --patience 7 \
      --num_workers 0 \
      --dataset_key mhealth \
      --stage 2 \
      --train_epochs 8 \
        > ./run_log/log_202512192033/mhealth/'model='$model_name'_finetune_'0.01.log 2>&1
  done