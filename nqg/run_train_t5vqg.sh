export CUDA_VISIBLE_DEVICES=1
rm -rvf t5vqgspt/test
MODEL=test
PRT_MODEL=google/t5-v1_1-small

python3 train_t5vqg.py \
  --model_name_or_path $PRT_MODEL \
  --tokenizer_name $PRT_MODEL \
  --config_name $PRT_MODEL \
  --output_dir t5vqgspt/$MODEL \
  --max_p_length 256 \
  --max_q_length 16 \
  --per_device_train_batch_size 8 \
  --evaluation_strategy 'steps' \
  --learning_rate 1e-3 \
  --optim adafactor \
  --lr_scheduler_type constant \
  --train_file /home/jhju/datasets/dragon.pseudo_datasets/dragon.colbertv2.pcentric.train.jsonl \
  --train_file /home/jhju/datasets/msmarco.triples_train_small/triples.train.small.v0.sample.jsonl \
  --max_steps 5000 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --freeze_t5 false \
  --pooling static \
  --n_soft_prompts 10 \
  --latent_size 128 \
  --k 0.00025 \
  --x0 1000 \
  --annealing logistic \
  --do_train \
  --do_eval 
  # --train_file /home/jhju/datasets/triples.train.small/triples.train.small.v1.jsonl \
