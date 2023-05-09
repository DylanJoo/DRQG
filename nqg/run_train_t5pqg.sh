export CUDA_VISIBLE_DEVICES=2

# VQG v0
python3 train_t5pqg.py \
  --model_name_or_path t5-base \
  --tokenizer_name t5-base \
  --config_name t5-base \
  --output_dir t5pqg \
  --max_length 256 \
  --per_device_train_batch_size 16 \
  --evaluation_strategy 'steps' \
  --train_file /home/jhju/datasets/triples.train.small/triples.train.small.v0.jsonl \
  --max_steps 10000 \
  --save_steps 2500 \
  --latent_size 256 \
  --k 0.00025 \
  --x0 2500 \
  --annealing logistic \
  --do_train \
  --do_eval
