TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl
k=4
gamma=1.0
tau=0.25

for ibn in qd dd na;do
    for cali in rank margin;do
        python3 train/train_softrelprompt.py \
            --model_name_or_path google/flan-t5-base \
            --tokenizer_name google/flan-t5-base \
            --config_name google/flan-t5-base \
            --output_dir models/checkpoint/calibrate_${cali}_ibn_${ibn} \
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
            --train_file ${TRAIN_FILE} \
            --instruction_prompt "Generate a question for the passage with relevance label: " \
            --relevant_prompt "true true true true true" \
            --irrelevant_prompt "false false false false false" \
            --do_train \
            --sample_random true  \
            --enable_similarity_loss $ibn \
            --tau $tau  \
            --enable_calibration $cali  \
            --calibration_margin_ngrams 1 2 \
            --gamma $gamma \
            --gradient_checkpointing true \
            --run_name prompt=5_batch=4_sample=top${k}_cali=${cali}_gamma=${gamma}_ibn=${ibn} > \
            models/checkpoint/calibrate_${cali}_ibn_${ibn}.log
    done
done

for model in calibrate_rank_ibn_qd calibrate_rank_ibn_dd calibrate_rank_ibn_na calibrate_margin_ibn_qd calibrate_margin_ibn_dd calibrate_margin_ibn_na;do
    bash run_generation.sh $model; bash run_evaluation.sh $model
done