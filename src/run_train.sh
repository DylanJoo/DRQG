BASE=bartqg
MODEL=nils

PRT_MODEL=google/flan-t5-base
PRT_CONFIG=google/flan-t5-base
SAVED_MODEL=models/checkpoint/flan-t5-base-qg
TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl

python3 train_vqg.py \
  --model_name_or_path $PRT_MODEL \
  --tokenizer_name $PRT_CONFIG \
  --config_name $PRT_CONFIG \
  --output_dir $SAVED_MODEL \
  --max_p_length 256 \
  --max_q_length 16 \
  --per_device_train_batch_size 4 \
  --m_negative_per_example 4 \
  --m_positive_per_example 4 \
  --learning_rate 1e-5 \
  --evaluation_strategy steps \
  --max_steps 10000 \
  --save_steps 2000 \
  --eval_steps 500 \
  --train_file $TRAIN_FILE \
  --latent_size 128 \
  --do_train \
  --do_eval 
