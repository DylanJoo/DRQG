TRAIN_FILE=/work/jhju/msmarco-psg.dcentric-minmax/ce.minilm.hardneg.vL.jsonl
MODEL_DIR=/work/jhju/flan-t5-readqg.msmarco-harg-neg
k=4
m=4
tau=0.25

python3 train/train_softrelprompt.py \
    --model_name_or_path google/flan-t5-base \
    --tokenizer_name google/flan-t5-base \
    --config_name google/flan-t5-base \
    --output_dir ${MODEL_DIR}/baseline_mle \
    --max_p_length 128 \
    --max_q_length 16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4\
    --m_positive_per_example 4 \
    --m_negative_per_example 4 \
    --learning_rate 1e-2 \
    --lr_scheduler_type constant \
    --max_steps 20000 \
    --save_steps 10000 \
    --train_file ${TRAIN_FILE} \
    --instruction_prompt "Generate a question for the passage with relevance label: " \
    --relevant_prompt "true true true true true" \
    --irrelevant_prompt "false false false false false" \
    --do_train \
    --sample_random true  \
    --sample_topk $k \
    --gradient_checkpointing true \
    --run_name ReadQG-baseline-mle
