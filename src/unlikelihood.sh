TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl
k=4
tau=0.25
for ibn in qd dd na;do
    python3 train/train_softrelprompt.py \
        --model_name_or_path google/flan-t5-base \
        --tokenizer_name google/flan-t5-base \
        --config_name google/flan-t5-base \
        --output_dir models/checkpoint/unlikelihood_ibn_${ibn} \
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
        --enable_similarity_loss ${ibn} \
        --tau $tau  \
        --enable_unlikelihood true \
        --gradient_checkpointing true \
        --run_name prompt=5_batch=4_sample=top${k}_ibn=${ibn}_tau=${tau} > ./models/checkpoint/unlikelihood_ibn_${ibn}.log
done

for model in unlikelihood_ibn_dd unlikelihood_ibn_qd unlikelihood_ibn_na; do
    bash run_generation.sh $model; bash run_evaluation.sh $model
done
