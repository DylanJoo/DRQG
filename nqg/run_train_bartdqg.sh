export CUDA_VISIBLE_DEVICES=1
BASE=bartdqg
MODEL=test

rm -rvf $BASE/$MODEL
# PRT_MODEL=facebook/bart-base
PRT_MODEL=bartqg-d2q/generalized/checkpoint-8000/
PRT_CONFIG=facebook/bart-base

# TRAIN_FILE=/home/jhju/datasets/dragon.pseudo_datasets/colbertv2.pcentric.train.v1.jsonl
# TRAIN_FILE=/home/jhju/datasets/dragon.pseudo_datasets/colbertv2.pcentric.train.vL.jsonl
TRAIN_FILE=/home/jhju/datasets/msmarco.triples_train_small/triples.train.small.vL.jsonl  

python3 train_dqg.py \
  --model_name_or_path $PRT_MODEL \
  --tokenizer_name $PRT_CONFIG \
  --config_name $PRT_CONFIG \
  --output_dir $BASE/$MODEL \
  --max_p_length 256 \
  --max_q_length 32 \
  --per_device_train_batch_size 4 \
  --m_samples_per_example 3 \
  --n_side 5 \
  --evaluation_strategy steps \
  --learning_rate 5e-4 \
  --train_file $TRAIN_FILE \
  --max_steps 10000 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --freeze_LM true \
  --freeze_embeds true \
  --freeze_a_layer true  \
  --freeze_cross_attn true \
  --pooling test \
  --add_attentive_pooler false \
  --n_soft_prompts 50 \
  --latent_size 128 \
  --k 0.0025 \
  --x0 2000 \
  --annealing logistic \
  --do_train \
  --do_eval 
