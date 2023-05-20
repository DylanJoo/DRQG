# MODEL=bartqg-d2q/relevant/
# MODEL=bartqg-d2q/generalized/
MODEL=bartqg-d2q/irrelevant/
mkdir -p $MODEL

export CUDA_VISIBLE_DEVICES=1
PRT_MODEL=facebook/bart-base
PRT_CONFIG=facebook/bart-base

python3 train_d2q.py \
  --model_name_or_path $PRT_MODEL \
  --tokenizer_name $PRT_MODEL \
  --config_name $PRT_CONFIG \
  --output_dir $MODEL \
  --max_p_length 256 \
  --max_q_length 32 \
  --learning_rate 1e-4 \
  --warmup_steps 4000 \
  --per_device_train_batch_size 64 \
  --train_file /home/jhju/datasets/msmarco.triples_train_small/triples.train.small.v0.jsonl \
  --irrelevant_included true \
  --relevant_included false \
  --max_steps 20000 \
  --save_steps 4000 \
  --do_train 
# relevant: /home/jhju/datasets/msmarco.triples_train_small/doc_query_pairs.train.jsonl 
# generalized: /home/jhju/datasets/msmarco.triples_train_small/triples.train.small.v0.jsonl
