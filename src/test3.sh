TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl
EVAL_FILE=data/ce.minilm.hardneg.vL.eval.small.jsonl

for k in 2;do
    for gamma in 0.5;do
        for tau in 0.25;do
            python3 train/train_softrelprompt.py \
                --model_name_or_path google/flan-t5-base \
                --tokenizer_name google/flan-t5-base \
                --config_name google/flan-t5-base \
                --output_dir models/checkpoint/prompt=5_batch=4_sample=top${k}_ngram=1+2_gamma=${gamma}_ibn=dw_tau=${tau}_mask=0.0_test=yesneg \
                --max_p_length 128 \
                --max_q_length 16 \
                --per_device_train_batch_size 4 \
                --per_device_eval_batch_size 4 \
                --m_positive_per_example 4 \
                --m_negative_per_example 4 \
                --learning_rate 1e-2 \
                --lr_scheduler_type constant \
                --evaluation_strategy steps \
                --max_steps 20000 \
                --save_steps 10000 \
                --eval_steps 500 \
                --train_file ${TRAIN_FILE} \
                --instruction_prompt "Generate a question for the passage with relevance label: " \
                --relevant_prompt "true true true true true" \
                --irrelevant_prompt "false false false false false" \
                --do_train \
                --do_eval \
                --enable_margin_gap_multivec f1  \
                --enable_margin_gap_multivec_ngrams 1 2 \
                --gamma $gamma \
                --enable_similarity_loss 'inbatch' \
                --document_wise_contrastive true \
                --random_corrupt_rate 0.0 \
                --tau $tau  \
                --sample_random true  \
                --sample_topk $k \
                --gradient_checkpointing false \
                --run_name prompt=5_batch=4_sample=top${k}_ngram=1+2_gamma=${gamma}_ibn=dw_tau=${tau}_mask=0.0-neg_test=yesneg
        done
    done
done
