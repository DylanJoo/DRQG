TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl
EVAL_FILE=data/ce.minilm.hardneg.vL.eval.small.jsonl

train_flan_baseline1:
	python3 train_baseline.py \
	  --model_name_or_path google/flan-t5-base \
	  --tokenizer_name google/flan-t5-base \
	  --config_name google/flan-t5-base \
	  --output_dir models/flan-t5-base-num \
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
	  --train_file ${TRAIN_FILE} \
	  --baseline_prefix "Generate a question with relevance score {0} for the passage: {1}" \
	  --do_train \
	  --do_eval 

train_flan_baseline2:
	python3 train_baseline.py \
	  --model_name_or_path google/flan-t5-base \
	  --tokenizer_name google/flan-t5-base \
	  --config_name google/flan-t5-base \
	  --output_dir models/flan-t5-base-extra \
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
	  --train_file ${TRAIN_FILE} \
	  --baseline_prefix "Generate a question with relevance label <extra_id_{0}> for the passage: {1}" \
	  --do_train \
	  --do_eval 

# Evaluate/generate the experimental results0
BASELINE1=models/flan-t5-base-qg-num/checkpoint-2000
generate_baseline1:
	# baseline model 1
	python3 generate.py \
	  --model_path ${BASELINE1} \
	  --model_name google/flan-t5-base \
	  --input_jsonl ${EVAL_FILE} \
	  --output_jsonl results/baseline1.jsonl \
	  --device cuda:1 \
	  --batch_size 8 \
	  --max_p_length 256 \
	  --num_pred 11 \
	  --prefix "Generate a question with relevance score {0} for the passage: {1}" 

BASELINE2=models/flan-t5-base-qg-extra/checkpoint-2000
generate_baseline2:
	python3 generate.py \
	  --model_path ${BASELINE2} \
	  --model_name google/flan-t5-base \
	  --input_jsonl ${EVAL_FILE} \
	  --output_jsonl results/baseline2.jsonl \
	  --device cuda:1 \
	  --batch_size 8 \
	  --max_p_length 256 \
	  --num_pred 11 \
	  --prefix "Generate a question with relevance label <extra_id_{0}> for the passage: {1}"
