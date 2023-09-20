TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl

python3 peft_prompt_tuning_rel1.py \
  --model_name_or_path google/flan-t5-base \
  --tokenizer_name google/flan-t5-base \
  --config_name google/flan-t5-base \
  --output_dir models/checkpoints/peft-prompt-tuning-rel1 \
  --max_p_length 256 \
  --max_q_length 16 \
  --per_device_train_batch_size 4 \
  --m_negative_per_example 2 \
  --m_positive_per_example 2 \
  --learning_rate 1 \
  --evaluation_strategy steps \
  --instruction_prompt "Given the relevance label, generate a question for the passage. relevance label: "  \
  --pos_neg_prompt "false true"  \
  --n_instruction_prompt 20 \
  --baseline_prefix "passage: {1}" \
  --max_steps 20000 \
  --save_steps 5000 \
  --eval_steps 500 \
  --train_file ${TRAIN_FILE} \
  --do_train \
  --do_eval 
  # --instruction_prompt "Given the relevance label, generate a question for the passage.\n relevance label: "  \
  # --pos_neg_prompt "false true"  \
  # --n_instruction_prompt 20 \
  # --baseline_prefix "\npassage: {1}" \
