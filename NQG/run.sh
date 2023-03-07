export CUDA_VISIBLE_DEVICES=0
python3 train.py \
  --model_name_or_path t5-small \
  --tokenizer_name t5-small \
  --config_name t5-small \
  --output_dir ./checkpoints/testing \
  --max_length 256 \
  --per_device_train_batch_size 4 \
  --evaluation_strategy 'steps' \
  --triplet temp/triples.train.small.sample.tsv \
  --max_steps 1000 \
  --save_steps 500 \
  --vae_latent_size 1024 \
  --vae_k 0.0025 \
  --vae_x0 5000 \
  --vae_annealing logistic \
  --do_train
