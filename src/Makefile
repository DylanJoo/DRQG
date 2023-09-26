TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl
EVAL_FILE=data/ce.minilm.hardneg.vL.eval.small.jsonl

train_softprompt_rel1:
	python3 train_old/train_softrelprompt.py \
	  --model_name_or_path google/flan-t5-base \
	  --tokenizer_name google/flan-t5-base \
	  --config_name google/flan-t5-base \
	  --output_dir models/checkpoints/flan-t5-base-rel1-un \
	  --max_p_length 128 \
	  --max_q_length 16 \
	  --per_device_train_batch_size 8 \
	  --per_device_eval_batch_size 8 \
	  --m_positive_per_example 2 \
	  --m_negative_per_example 2 \
	  --learning_rate 1e-2 \
	  --lr_scheduler_type constant \
	  --evaluation_strategy steps \
	  --max_steps 20000 \
	  --save_steps 5000 \
	  --eval_steps 500 \
	  --train_file ${TRAIN_FILE} \
	  --instruction_prompt "Generate a question for the passage with relevance label: " \
	  --relevant_prompt "true" \
	  --irrelevant_prompt "false" \
	  --do_train \
	  --do_eval \
	  --enable_unlikelihood true

train_softprompt_rel2:
	python3 train_old/train_softrelprompt.py \
	  --model_name_or_path google/flan-t5-base \
	  --tokenizer_name google/flan-t5-base \
	  --config_name google/flan-t5-base \
	  --output_dir models/checkpoints/flan-t5-base-rel2-un \
	  --max_p_length 112 \
	  --max_q_length 16 \
	  --per_device_train_batch_size 8 \
	  --per_device_eval_batch_size 8 \
	  --m_positive_per_example 2 \
	  --m_negative_per_example 2 \
	  --learning_rate 1e-2 \
	  --lr_scheduler_type constant \
	  --evaluation_strategy steps \
	  --max_steps 20000 \
	  --save_steps 5000 \
	  --eval_steps 500 \
	  --train_file ${TRAIN_FILE} \
	  --instruction_prompt "Generate a question for the passage with relevance labels: " \
	  --relevant_prompt "true true true true true true true true true true" \
	  --irrelevant_prompt "false false false false false false false false false false" \
	  --do_train \
	  --do_eval  \
	  --enable_unlikelihood true

# train_baseline:
# 	python3 train_old/train_baseline.py \
# 	  --model_name_or_path google/flan-t5-base  \
# 	  --tokenizer_name google/flan-t5-base \
# 	  --config_name google/flan-t5-base \
# 	  --output_dir models/checkpoints/flan-t5-base-num-prefix \
# 	  --max_p_length 256 \
# 	  --max_q_length 16 \
# 	  --per_device_train_batch_size 8 \
# 	  --per_device_eval_batch_size 8 \
# 	  --m_negative_per_example 2 \
# 	  --m_positive_per_example 2 \
# 	  --learning_rate 1e-4 \
# 	  --evaluation_strategy steps \
# 	  --max_steps 20000 \
# 	  --save_steps 2000 \
# 	  --eval_steps 500 \
# 	  --train_file ${TRAIN_FILE} \
# 	  --baseline_prefix "Generate a question with relevance score {0} for the passage: {1}" \
# 	  --prefix_tuning true \
# 	  --do_train \
# 	  --do_eval 

# train_softprompt_rel_doc:
# 	python3 train_old/train_softprompt_rel_doc.py \
# 	  --model_name_or_path google/flan-t5-base \
# 	  --tokenizer_name google/flan-t5-base \
# 	  --config_name google/flan-t5-base \
# 	  --output_dir models/checkpoints/flan-t5-base-soft-reldoc \
# 	  --max_p_length 256 \
# 	  --max_q_length 16 \
# 	  --per_device_train_batch_size 8 \
# 	  --per_device_eval_batch_size 8 \
# 	  --m_positive_per_example 2 \
# 	  --m_negative_per_example 2 \
# 	  --learning_rate 1e-3 \
# 	  --evaluation_strategy steps \
# 	  --max_steps 20000 \
# 	  --save_steps 5000 \
# 	  --eval_steps 500 \
# 	  --train_file ${TRAIN_FILE} \
# 	  --instruction_prompt "Generate a question based on semantic vectors of the relevance and a document. relevance: " \
# 	  --relevance_prompt "true true true true true true true true true true" \
# 	  --irrelevance_prompt "false false false false false false false false false false" \
# 	  --baseline_prefix "document: {1}" \
# 	  --random_init false \
# 	  --do_train \
# 	  --do_eval
