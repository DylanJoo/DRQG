export CUDA_VISIBLE_DEVICES=1

python3 train_dev.py \
  --model_name_or_path t5-base \
  --tokenizer_name t5-base \
  --config_name t5-base \
  --output_dir t5vqgdev\
  --max_length 128 \
  --per_device_train_batch_size 8 \
  --evaluation_strategy 'steps' \
  --train_file /home/jhju/datasets/triples.train.small/triples.train.small.v0.jsonl \
  --max_steps 5000 \
  --save_steps 2500 \
  --n_soft_prompts 1 \
  --latent_size 128 \
  --k 0.00025 \
  --x0 1000 \
  --annealing logistic \
  --do_train \
  --do_eval 
