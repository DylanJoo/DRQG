TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl

train_flan_baseline:
	python3 train_vqg.py \
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
	  --latent_size 128 \
	  --do_train \
	  --do_eval 
