export CUDA_VISIBLE_DEVICES=1

# VQG: Use the variatinoal token as residual learning
# VQG (debug1): inplace switch the dummy token directly
# VQG (debug2): KL weight == 1
# VQG (debug3): add projection layers
python3 train.py \
  --model_name_or_path t5-base \
  --tokenizer_name t5-base \
  --config_name t5-base \
  --output_dir t5vqgv1-debug-3\
  --max_length 128 \
  --per_device_train_batch_size 4 \
  --evaluation_strategy 'steps' \
  --train_file /home/jhju/datasets/triples.train.small/triples.train.small.v0.jsonl \
  --max_steps 5000 \
  --save_steps 2500 \
  --latent_size 128 \
  --k 0.00025 \
  --x0 1000 \
  --annealing logistic \
  --do_train \
  --do_eval 
