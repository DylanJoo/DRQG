export CUDA_VISIBLE_DEVICES=0
BASE=bartvqgspt
MODEL=colbert-warm-ada_50-Z_128-BS_12

rm -rvf $BASE/$MODEL
PRT_MODEL=bartqg-d2q/checkpoint-8000/
PRT_CONFIG=facebook/bart-base

TRAIN_FILE=/home/jhju/datasets/dragon.pseudo_datasets/colbertv2.pcentric.train.v1.jsonl

python3 train_vqg.py \
  --model_name_or_path $PRT_MODEL \
  --tokenizer_name $PRT_CONFIG \
  --config_name $PRT_CONFIG \
  --output_dir $BASE/$MODEL \
  --max_p_length 256 \
  --max_q_length 16 \
  --per_device_train_batch_size 12 \
  --m_samples_per_example 1 \
  --n_side 5 \
  --evaluation_strategy steps \
  --learning_rate 1e-3 \
  --lr_scheduler_type constant \
  --train_file $TRAIN_FILE \
  --max_steps 10000 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --freeze_LM true \
  --freeze_embeds false \
  --pooling static \
  --n_soft_prompts 1 \
  --latent_size 128 \
  --k 0.5 \
  --x0 2000 \
  --annealing logistic \
  --do_train \
  --do_eval 
