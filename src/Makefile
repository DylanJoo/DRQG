# ---  Model settings --- #
# [*] baseline1: relevance as number as prompt
# 	* full fine-tuning
# 	* prefix tuning
# baseline2: relevance as extra id label as prompt
# 	* full fine-tuning
# 	* prefix tuning
# d2q: No negative samples (all positive)
# 	* soft prompting
# ----------------------- #

TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl
EVAL_FILE=data/ce.minilm.hardneg.vL.eval.small.jsonl

SIZE=base
train_baseline1:
	export CUDA_VISIBLE_DEVICES=2; python3 train_baseline.py \
	  --model_name_or_path google/flan-t5-${SIZE} \
	  --tokenizer_name google/flan-t5-${SIZE} \
	  --config_name google/flan-t5-${SIZE} \
	  --output_dir models/checkpoints/flan-t5-${SIZE}-num-prefix \
	  --max_p_length 256 \
	  --max_q_length 16 \
	  --per_device_train_batch_size 4 \
	  --m_negative_per_example 2 \
	  --m_positive_per_example 2 \
	  --learning_rate 1e-4 \
	  --evaluation_strategy steps \
	  --max_steps 10000 \
	  --save_steps 2000 \
	  --eval_steps 500 \
	  --train_file ${TRAIN_FILE} \
	  --baseline_prefix "Generate a question with relevance score {0} for the passage: {1}" \
	  --prefix_tuning true \
	  --do_train \
	  --do_eval 

generate_baseline1:
	# baseline model 1
	python3 generate.py \
	  --model_path models/checkpoints/flan-t5-base-num-prefix/checkpoint-2000 \
	  --model_name google/flan-t5-base \
	  --input_jsonl ${EVAL_FILE} \
	  --output_jsonl results/baseline1_prefix.jsonl \
	  --device cuda:1 \
	  --batch_size 8 \
	  --max_p_length 256 \
	  --num_pred 11 \
	  --prefix "Generate a question with relevance score {0} for the passage: {1}" 

generate_baseline1-large:
	python3 generate.py \
	  --model_path models/checkpoints/flan-t5-large-num-prefix/checkpoint-4000 \
	  --model_name google/flan-t5-large \
	  --input_jsonl ${EVAL_FILE} \
	  --output_jsonl results/baseline1_large_prefix.jsonl \
	  --device cuda:1 \
	  --batch_size 4 \
	  --max_p_length 256 \
	  --num_pred 11 \
	  --prefix "Generate a question with relevance score {0} for the passage: {1}" 


# [TO BE REMOVE, JUST A TEST] soft prompt (instruction N dim; i.e., d2q)
# train_softprompt_d2q:
# 	export CUDA_VISIBLE_DEVICES=1 
# 	python3 train_softprompt.py \
# 	  --model_name_or_path google/flan-t5-large \
# 	  --tokenizer_name google/flan-t5-large \
# 	  --config_name google/flan-t5-large \
# 	  --output_dir models/checkpoints/flan-t5-large-soft-d2q \
# 	  --max_p_length 256 \
# 	  --max_q_length 16 \
# 	  --per_device_train_batch_size 16 \
# 	  --m_positive_per_example 1 \
# 	  --m_negative_per_example 0 \
# 	  --learning_rate 1e-4 \
# 	  --evaluation_strategy steps \
# 	  --max_steps 10000 \
# 	  --save_steps 4000 \
# 	  --eval_steps 500 \
# 	  --train_file ${TRAIN_FILE} \
# 	  --instruction_prompt "Generate a question for the passage: " \
# 	  --do_train \
# 	  --do_eval 

# generate_baseline_d2q:
# 	python3 generate.py \
# 	  --model_path models/checkpoints/flan-t5-large-soft-d2q/checkpoint-4000 \
# 	  --model_name google/flan-t5-large \
# 	  --input_jsonl ${EVAL_FILE} \
# 	  --output_jsonl results/baseline_d2q_prefix.jsonl \
# 	  --device cuda:1 \
# 	  --batch_size 4 \
# 	  --max_p_length 256 \
# 	  --num_pred 11 \
# 	  --instruction_prompt "Generate a question for the passage: " 


# soft prompt (instruction N dim + relevance 1 dim)
train_softprompt_rel:
	export CUDA_VISIBLE_DEVICES=1; python3 train_softprompt.py \
	  --model_name_or_path google/flan-t5-base \
	  --tokenizer_name google/flan-t5-base \
	  --config_name google/flan-t5-base \
	  --output_dir models/checkpoints/flan-t5-base-soft-rel \
	  --max_p_length 256 \
	  --max_q_length 16 \
	  --per_device_train_batch_size 8 \
	  --m_positive_per_example 2 \
	  --m_negative_per_example 2 \
	  --learning_rate 1e-3 \
	  --evaluation_strategy steps \
	  --max_steps 10000 \
	  --save_steps 4000 \
	  --eval_steps 500 \
	  --train_file ${TRAIN_FILE} \
	  --instruction_prompt "Generate a question for the passage with relevance label: " \
	  --relevance_prompt "false true" \
	  --do_train \
	  --do_eval 

generate_softprompt_rel:
	python3 generate.py \
	  --model_path models/checkpoints/flan-t5-base-soft-rel/checkpoint-4000 \
	  --model_name google/flan-t5-base \
	  --input_jsonl ${EVAL_FILE} \
	  --output_jsonl results/baseline_softprompt.jsonl \
	  --device cuda:2 \
	  --batch_size 8 \
	  --max_p_length 256 \
	  --num_pred 11 \
	  --instruction_prompt "Generate a question for the passage with relevance label: " \
	  --relevance_prompt "false true"
