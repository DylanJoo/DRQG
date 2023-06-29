BASE=bart-relqg
MODEL=$BASE/test
rm -rvf $MODEL/*

export CUDA_VISIBLE_DEVICES=1
PRT_MODEL=bart-d2q/relevant/checkpoint-16000
PRT_CONFIG=facebook/bart-base
TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl
# TRAIN_FILE=/home/jhju/datasets/redragon.pseudo_datasets/colbertv2.pcentric.train.vL.jsonl

python3 train_qg.py \
  --model_name_or_path $PRT_MODEL \
  --tokenizer_name $PRT_CONFIG \
  --config_name $PRT_CONFIG \
  --output_dir $BASE/$MODEL \
  --max_p_length 200 \
  --max_q_length 16 \
  --per_device_train_batch_size 4 \
  --m_negative_per_example 4 \
  --m_positive_per_example 4 \
  --learning_rate 1e-3 \
  --lr_scheduler_type constant \
  --evaluation_strategy steps \
  --max_steps 10000 \
  --save_steps 2000 \
  --eval_steps 500 \
  --train_file $TRAIN_FILE \
  --latent_size 128 \
  --add_classification_head true \
  --pos_anchors 'true true true true true' \
  --neg_anchors 'false false false false false' \
  --pooling mean \
  --activation tanh \
  --warmup_ratio 0.1 \
  --do_train \
  --do_eval

  # --annealing_fn cyclic \
  # --n_cycle 4 \

# [NOTE] 
# larger leanring rate 
# longer and diversified anchors  
# n cycle  
# larger negative samples  

# BASE=bartqg
# MODEL=$BASE/mean/
#
# export CUDA_VISIBLE_DEVICES=0
# PRT_MODEL=bartqg-d2q/mean/checkpoint-16000
# PRT_CONFIG=facebook/bart-base
# TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl
#
# python3 train_vqg.py \
#   --model_name_or_path $PRT_MODEL \
#   --tokenizer_name $PRT_CONFIG \
#   --config_name $PRT_CONFIG \
#   --output_dir $MODEL \
#   --max_p_length 200 \
#   --max_q_length 16 \
#   --per_device_train_batch_size 4 \
#   --m_negative_per_example 4 \
#   --m_positive_per_example 4 \
#   --learning_rate 1e-3 \
#   --lr_scheduler_type constant \
#   --evaluation_strategy steps \
#   --max_steps 10000 \
#   --save_steps 2000 \
#   --eval_steps 500 \
#   --train_file $TRAIN_FILE \
#   --latent_size 128 \
#   --add_classification_head true \
#   --prompts 'what where when who how why which' \
#   --label_prompts 'false true' \
#   --pooling mean \
#   --activation tanh \
#   --warmup_ratio 0.1 \
#   --do_train \
#   --do_eval 

