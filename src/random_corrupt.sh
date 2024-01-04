TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl
EVAL_FILE=data/ce.minilm.hardneg.vL.eval.small.jsonl

bs=4
let m=32/2/$bs
for rate in 0.10 0.25 0.50;do
    python3 train/train_softrelprompt.py \
        --model_name_or_path google/flan-t5-base \
        --tokenizer_name google/flan-t5-base \
        --config_name google/flan-t5-base \
        --output_dir models/checkpoint/random_corrupt_${rate}_bs${bs}_top${k} \
        --max_p_length 128 \
        --max_q_length 16 \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --m_positive_per_example $m \
        --m_negative_per_example $m \
        --learning_rate 1e-2 \
        --lr_scheduler_type constant \
        --max_steps 20000 \
        --save_steps 5000 \
        --eval_steps 500 \
        --train_file ${TRAIN_FILE} \
        --instruction_prompt "Generate a question for the passage with relevance label: " \
        --relevant_prompt "true true true true true" \
        --irrelevant_prompt "false false false false false" \
        --do_train \
        --sample_random true  \
        --sample_topk $k \
        --random_corrupt_rate $rate \
        --gradient_checkpointing true \
        --run_name prompt=5_batch=${bs}_sample=top${k} > ./models/checkpoint/random_corrupt_${rate}_bs${bs}_top${k}.log
done

for model in random_corrupt_0.10_bs4_top1 random_corrupt_0.25_bs4_top2 random_corrupt_0.50_bs4_top4; do
    bash run_generation.sh $model; bash run_evaluation.sh $model
done
