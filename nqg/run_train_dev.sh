export CUDA_VISIBLE_DEVICES=2
rm -rvf t5vqgspt/BM25-0_P-1_Z-128_BS-4-NONE/

python3 train_dev.py \
  --model_name_or_path t5-base \
  --tokenizer_name t5-base \
  --config_name t5-base \
  --output_dir t5vqgspt/BM25-0_P-20_Z-128_BS-4  \
  --max_p_length 256 \
  --max_q_length 16 \
  --per_device_train_batch_size 4 \
  --evaluation_strategy 'steps' \
  --optim adafactor \
  --learning_rate 3e-3 \
  --train_file /home/jhju/datasets/triples.train.small/triples.train.small.v0.jsonl \
  --max_steps 10000 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --n_soft_prompts 20 \
  --latent_size 128 \
  --k 0.0025 \
  --x0 1000 \
  --annealing 'logistic' \
  --do_train \
  --do_eval 
