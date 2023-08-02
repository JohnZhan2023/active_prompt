# Run inference with selected demos as prompts
python inference.py --random_seed=42 --dataset="gsm8k" --model="text-davinci-002" \
    --prompt_source_path="./dataset/GSM8K/train.jsonl" --prompt_num_path="./logdifference_results/gsm8k_baichuan7b_8-1_trainsplit-val.txt" \
    --max_length_cot=256 --qes_limit=100 --api_time_interval=16 --multipath=1