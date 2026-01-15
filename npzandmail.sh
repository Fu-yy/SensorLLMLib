python /root/autodl-tmp/SensorLLMLib/data_provider/data_loader.py --build_npz --npz_target capture24 --npz_out ./npz_cache --capture24_cache_mode pid --npz_workers 10 --npz_compressed 0   --seq_len 500 --stride 250
echo $?
python /root/autodl-tmp/SensorLLMLib/send_email.py