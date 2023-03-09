# batch size 8 (in fact 16) used 15G GPU
export CUDA_VISIBLE_DEVICES=1,2
python3 train.py \
  --model_name_or_path t5-base \
  --tokenizer_name t5-base \
  --config_name t5-base \
  --output_dir ./checkpoints/testing \
  --max_length 256 \
  --per_device_train_batch_size 8 \
  --triplet /tmp2/jhju/datasets/triples.train.small.tsv \
  --max_steps 10000 \
  --save_steps 2500 \
  --latent_size 256 \
  --k 0.00025 \
  --x0 2500 \
  --annealing logistic \
  --do_train
