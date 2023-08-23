export CUDA_VISIBLE_DEVICES=1
rm -rvf t5vqgspt/test
MODEL=test
PRT_MODEL=google/t5-v1_1-small
PRT_CONFIG=google/t5-v1_1-small
# PRT_MODEL=doc2query/msmarco-t5-small-v1

python3 train_t5vqg.py \
  --model_name_or_path $PRT_MODEL \
  --tokenizer_name $PRT_CONFIG \
  --config_name $PRT_CONFIG \
  --output_dir t5vqgspt/$MODEL \
  --max_p_length 256 \
  --max_q_length 16 \
  --per_device_train_batch_size 8 \
  --evaluation_strategy 'steps' \
  --learning_rate 1e-3 \
  --optim adafactor \
  --lr_scheduler_type constant \
  --train_file /home/jhju/datasets/dragon.pseudo_datasets/dragon.colbertv2.pcentric.train.jsonl \
  --max_steps 5000 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --freeze_LM true \
  --pooling adaptive \
  --n_soft_prompts 1 \
  --latent_size 128 \
  --k 0.00025 \
  --x0 2000 \
  --annealing logistic \
  --do_train \
  --do_eval 
  # --train_file /home/jhju/datasets/triples.train.small/triples.train.small.v1.jsonl \
