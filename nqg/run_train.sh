export CUDA_VISIBLE_DEVICES=1
BASE=bartvqg
MODEL=colbert-cold-ada_10-B_4x3-full

rm -rvf $BASE/$MODEL
# PRT_MODEL=facebook/bart-base
PRT_MODEL=bartqg-d2q/relevant/checkpoint-16000
PRT_CONFIG=facebook/bart-base

TRAIN_FILE=/home/jhju/datasets/dragon.pseudo_datasets/colbertv2.pcentric.train.vL.jsonl
# TRAIN_FILE=/home/jhju/datasets/msmarco.triples_train_small/triples.train.small.vL.jsonl  

python3 train_vqg.py \
  --model_name_or_path $PRT_MODEL \
  --tokenizer_name $PRT_CONFIG \
  --config_name $PRT_CONFIG \
  --output_dir $BASE/$MODEL \
  --max_p_length 256 \
  --max_q_length 16 \
  --per_device_train_batch_size 4 \
  --m_negative_per_example 3 \
  --n_side 5 \
  --evaluation_strategy steps \
  --learning_rate 1e-3 \
  --lr_scheduler_type constant \
  --train_file $TRAIN_FILE \
  --max_steps 10000 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --pooling attentive \
  --add_attentive_pooler true \
  --n_soft_prompts 10 \
  --latent_size 128 \
  --has_compressed_layer false \
  --k 0.025 \
  --x0 2000 \
  --warmup_ratio 0.1 \
  --annealing logistic \
  --do_train \
  --do_eval 
  # --freeze_LM true \
  # --freeze_embeds true \
  # --freeze_a_layer true  \
  # --freeze_cross_attn true \
