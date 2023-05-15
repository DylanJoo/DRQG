export CUDA_VISIBLE_DEVICES=1
MODEL=colbert-stat_10-Z_128-BS_4
rm -rvf bartvqgspt/$MODEL
PRT_MODEL=facebook/bart-base
PRT_CONFIG=facebook/bart-base

python3 train_vqg.py \
  --model_name_or_path $PRT_MODEL \
  --tokenizer_name $PRT_CONFIG \
  --config_name $PRT_CONFIG \
  --output_dir bartvqgspt/$MODEL \
  --max_p_length 256 \
  --max_q_length 16 \
  --per_device_train_batch_size 4 \
  --m_samples_per_example 1 \
  --n_side 5 \
  --evaluation_strategy steps \
  --learning_rate 2e-5 \
  --lr_scheduler_type constant \
  --train_file /home/jhju/datasets/dragon.pseudo_datasets/colbertv2.pcentric.train.v1.jsonl \
  --max_steps 10000 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --freeze_LM true \
  --freeze_embeds false \
  --warmup_steps 1000 \
  --pooling static \
  --n_soft_prompts 10 \
  --latent_size 128 \
  --k 0.5 \
  --x0 1000 \
  --annealing logistic \
  --do_train \
  --do_eval 
  # --random_masking_ratio 0.0 \
  # --train_file /home/jhju/datasets/msmarco.triples_train_small/triples.train.small.v1.jsonl \
