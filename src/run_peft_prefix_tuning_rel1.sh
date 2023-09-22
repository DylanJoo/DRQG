TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl

python3 peft_prefix_tuning_rel1.py \
  --model_name_or_path google/flan-t5-base \
  --tokenizer_name google/flan-t5-base \
  --config_name google/flan-t5-base \
  --output_dir models/checkpoints/peft-prefix-tuning-rel1 \
  --max_p_length 256 \
  --max_q_length 16 \
  --per_device_train_batch_size 4 \
  --m_negative_per_example 2 \
  --m_positive_per_example 2 \
  --learning_rate 1e-3 \
  --evaluation_strategy steps \
  --n_instruction_prompt 10 \
  --baseline_prefix "{1}" \
  --pos_neg_prompt "false true"  \
  --max_steps 20000 \
  --save_steps 2000 \
  --eval_steps 500 \
  --train_file ${TRAIN_FILE} \
  --do_train \
  --do_eval 
