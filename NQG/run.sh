export CUDA_VISIBLE_DEVICES=0
python3 train.py \
  --model_name_or_path t5-base \
  --tokenizer_name t5-base \
  --config_name t5-base \
  --output_dir ./checkpoints/testing \
  --max_length 256 \
  --per_device_train_batch_size 16 \
  --evaluation_strategy 'steps' \
  --triplet temp/triples.train.small.sample.tsv \
  --max_steps 1000 \
  --save_steps 500 \
  --latent_size 256 \
  --k 0.0025 \
  --x0 5000 \
  --annealing logistic \
  --do_train
