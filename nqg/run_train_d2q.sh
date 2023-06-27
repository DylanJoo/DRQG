# BASE=bartqg-d2q
# MODEL=$BASE/mean/
#
# export CUDA_VISIBLE_DEVICES=0
# PRT_MODEL=facebook/bart-base
# PRT_CONFIG=facebook/bart-base
# TRAIN_FILE=/home/jhju/datasets/msmarco.triples_train_small/doc_query_pairs.train.jsonl
#
# python3 train_d2q.py \
#   --model_name_or_path $PRT_MODEL \
#   --tokenizer_name $PRT_MODEL \
#   --config_name $PRT_CONFIG \
#   --output_dir $MODEL \
#   --max_p_length 256 \
#   --max_q_length 32 \
#   --learning_rate 1e-4 \
#   --warmup_steps 4000 \
#   --per_device_train_batch_size 64 \
#   --train_file $TRAIN_FILE \
#   --irrelevant_included false \
#   --relevant_included true \
#   --max_steps 20000 \
#   --save_steps 4000 \
#   --do_train 

# BASE=bartqg-d2q
# MODEL=$BASE/mean/
#
# export CUDA_VISIBLE_DEVICES=0
# PRT_MODEL=facebook/bart-base
# PRT_CONFIG=facebook/bart-base
# TRAIN_FILE=/home/jhju/datasets/msmarco.triples_train_small/doc_query_pairs.train.jsonl
#
# python3 train_d2q.py \
#   --model_name_or_path $PRT_MODEL \
#   --tokenizer_name $PRT_MODEL \
#   --config_name $PRT_CONFIG \
#   --output_dir $MODEL \
#   --pooling mean \
#   --max_p_length 256 \
#   --max_q_length 32 \
#   --learning_rate 1e-4 \
#   --warmup_steps 4000 \
#   --per_device_train_batch_size 64 \
#   --train_file $TRAIN_FILE \
#   --irrelevant_included false \
#   --relevant_included true \
#   --max_steps 20000 \
#   --save_steps 4000 \
#   --do_train

BASE=maskqg-d2q
MODEL=$BASE/halfhalf/

export CUDA_VISIBLE_DEVICES=0
PRT_MODEL=facebook/bart-base
PRT_CONFIG=facebook/bart-base
# TRAIN_FILE=/home/jhju/datasets/msmarco.triples_train_small/doc_query_pairs.train.jsonl
TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl

python3 train_d2q.py \
  --model_name_or_path $PRT_MODEL \
  --tokenizer_name $PRT_MODEL \
  --config_name $PRT_CONFIG \
  --output_dir $MODEL \
  --pooling mean \
  --max_p_length 512 \
  --max_q_length 32 \
  --learning_rate 1e-4 \
  --warmup_steps 4000 \
  --per_device_train_batch_size 4 \
  --m_negative_per_example 4 \
  --m_positive_per_example 4 \
  --train_file $TRAIN_FILE \
  --irrelevant_included false \
  --relevant_included true \
  --max_steps 20000 \
  --save_steps 4000 \
  --do_train 
