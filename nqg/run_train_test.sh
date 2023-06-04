export CUDA_VISIBLE_DEVICES=0
BASE=bartcvqg
MODEL=nils-warm-mean-444-hard-temp-5w1h

rm -rvf $BASE/$MODEL
# PRT_MODEL=facebook/bart-base
PRT_MODEL=bartqg-d2q/relevant/checkpoint-16000
PRT_CONFIG=facebook/bart-base

# TRAIN_FILE=/home/jhju/datasets/redragon.pseudo_datasets/colbertv2.pcentric.train.vL.jsonl
TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl

python3 train_vqg_test.py \
  --model_name_or_path $PRT_MODEL \
  --tokenizer_name $PRT_CONFIG \
  --config_name $PRT_CONFIG \
  --output_dir $BASE/$MODEL \
  --max_p_length 200 \
  --max_q_length 16 \
  --per_device_train_batch_size 4 \
  --m_negative_per_example 4 \
  --m_positive_per_example 4 \
  --evaluation_strategy steps \
  --learning_rate 1e-3 \
  --train_file $TRAIN_FILE \
  --max_steps 10000 \
  --save_steps 2000 \
  --eval_steps 500 \
  --add_classification_head true \
  --num_labels 2 \
  --pooling 'mean' \
  --latent_size 128 \
  --has_compressed_layer true \
  --freeze_LM true \
  --n_prompts 10 \
  --used_prompt 'generate positive or negative question for this passage' \
  --n_labels 2 \
  --used_label 'false true' \
  --warmup_ratio 0.2 \
  --annealing cyclic \
  --do_train \
  --do_eval 
