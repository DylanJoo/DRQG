TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl

# use the numeric string token as condition
train_flan_baseline1:
	python3 train_baseline.py \
	  --model_name_or_path google/flan-t5-base \
	  --tokenizer_name google/flan-t5-base \
	  --config_name google/flan-t5-base \
	  --output_dir models/flan-t5-base-qg \
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
	  --latent_size 128 \
	  --do_train \
	  --do_eval 

# use the special token as condition
train_flan_baseline2:
	python3 train_baseline.py \
	  --model_name_or_path google/flan-t5-base \
	  --tokenizer_name google/flan-t5-base \
	  --config_name google/flan-t5-base \
	  --output_dir models/flan-t5-base-qg \
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
	  --latent_size 128 \
	  --do_train \
	  --do_eval 
