TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl
k=4
gamma=0.5
tau=0.25
python3 train/train_softrelprompt.py \
    --model_name_or_path google/flan-t5-base \
    --tokenizer_name google/flan-t5-base \
    --config_name google/flan-t5-base \
    --output_dir models/checkpoint/unlikelihood_w_ibn \
    --max_p_length 128 \
    --max_q_length 16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --m_positive_per_example 4 \
    --m_negative_per_example 4 \
    --learning_rate 1e-2 \
    --lr_scheduler_type constant \
    --max_steps 20000 \
    --save_steps 10000 \
    --eval_steps 500 \
    --train_file ${TRAIN_FILE} \
    --instruction_prompt "Generate a question for the passage with relevance label: " \
    --relevant_prompt "true true true true true" \
    --irrelevant_prompt "false false false false false" \
    --do_train \
    --sample_random true  \
    --sample_topk $k \
    --enable_similarity_loss 'inbatch' \
    --tau $tau  \
    --enable_unlikelihood true \
    --gradient_checkpointing true \
    --run_name prompt=5_batch=4_sample=top${k}_ibn=allw_tau=${tau} > ./models/checkpoint/log_unlikelihood_w_ibn

python3 train/train_softrelprompt.py \
    --model_name_or_path google/flan-t5-base \
    --tokenizer_name google/flan-t5-base \
    --config_name google/flan-t5-base \
    --output_dir models/checkpoint/unlikelihood_wo_ibn \
    --max_p_length 128 \
    --max_q_length 16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --m_positive_per_example 4 \
    --m_negative_per_example 4 \
    --learning_rate 1e-2 \
    --lr_scheduler_type constant \
    --max_steps 20000 \
    --save_steps 10000 \
    --eval_steps 500 \
    --train_file ${TRAIN_FILE} \
    --instruction_prompt "Generate a question for the passage with relevance label: " \
    --relevant_prompt "true true true true true" \
    --irrelevant_prompt "false false false false false" \
    --do_train \
    --sample_random true  \
    --sample_topk $k \
    --enable_unlikelihood true \
    --gradient_checkpointing true \
    --run_name prompt=5_batch=4_sample=top${k}_ngram=1+2_ibn=na > ./models/checkpoint/log_unlikelihood_wo_ibn

for model in unlikelihood_w_ibn unlikelihood_wo_ibn; do
    bash run_generation.sh $model; bash run_evaluation.sh $model
done
