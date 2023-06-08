export CUDA_VISIBLE_DEVICES=0

BASE=bart-option1
MODEL=nils

rm -rvf $BASE/$MODEL
# PRT_MODEL=facebook/bart-base
PRT_MODEL=bartqg-d2q/relevant/checkpoint-16000
PRT_CONFIG=facebook/bart-base

TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl
python3 train_vqg.py \
  --model_name_or_path $PRT_MODEL \
  --tokenizer_name $PRT_CONFIG \
  --config_name $PRT_CONFIG \
  --output_dir $BASE/$MODEL \
  --max_p_length 200 \
  --max_q_length 16 \
  --per_device_train_batch_size 4 \
  --m_negative_per_example 4 \
  --m_positive_per_example 4 \
  --n_side 5 \
  --evaluation_strategy steps \
  --learning_rate 1e-3 \
  --lr_scheduler_type constant \
  --train_file $TRAIN_FILE \
  --max_steps 10000 \
  --save_steps 2000 \
  --eval_steps 500 \
  --pooling adaptive \
  --add_classification_head true \
  --adaptive_pooling mean \
  --n_soft_prompts 10 \
  --latent_size 128 \
  --has_compressed_layer true \
  --initialize_from_vocab true \
  --used_prompt 'generate positive or negative question for this passage' \
  --warmup_ratio 0.1 \
  --annealing cyclic \
  --n_cycle 10 \
  --do_train \
  --do_eval 
