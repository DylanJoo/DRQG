export CUDA_VISIBLE_DEVICES=1
BASE=bartvqg-test-qp
MODEL=redragon-warm-adamean-B_811

rm -rvf $BASE/$MODEL
PRT_MODEL=bartqg-d2q/relevant/checkpoint-8000
# PRT_MODEL=facebook/bart-base
PRT_CONFIG=facebook/bart-base

# TRAIN_FILE=/home/jhju/datasets/redragon.pseudo_datasets/colbertv2.pcentric.train.vL.jsonl
TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl
# TRAIN_FILE=/home/jhju/datasets/msmarco.triples_train_small/triples.train.small.vL.jsonl  
python3 train_vqg_dev.py \
  --model_name_or_path $PRT_MODEL \
  --tokenizer_name $PRT_CONFIG \
  --config_name $PRT_CONFIG \
  --output_dir $BASE/$MODEL \
  --max_p_length 256 \
  --max_q_length 20 \
  --per_device_train_batch_size 8 \
  --m_negative_per_example 5 \
  --m_positive_per_example 1 \
  --n_side 5 \
  --evaluation_strategy steps \
  --learning_rate 1e-3 \
  --lr_scheduler_type linear \
  --train_file $TRAIN_FILE \
  --max_steps 20000 \
  --save_steps 2000 \
  --eval_steps 500 \
  --pooling adaptive \
  --adaptive_pooling mean \
  --add_attentive_pooler false \
  --n_soft_prompts 10 \
  --latent_size 128 \
  --has_compressed_layer true \
  --initialize_from_vocab true \
  --add_classification_head false \
  --used_prompt 'generate positive or negative question based on this passage' \
  --disable_dropout false \
  --k 0.025 \
  --x0 1000 \
  --warmup_ratio 0.1 \
  --annealing logistic \
  --do_train \
  --do_eval 
