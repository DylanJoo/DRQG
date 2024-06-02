export CUDA_VISIBLE_DEVICES=0

TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl
EVAL_FILE=data/ce.minilm.hardneg.vL.eval.small.jsonl
MODEL_DIR=/work/jhju/readqg-baseline
k=4
gamma=1.0
tau=0.25

for ibn in dd;do
    for cali in rank;do
        python3 train/train_softrelprompt.py \
            --model_name_or_path google/flan-t5-base \
            --tokenizer_name google/flan-t5-base \
            --config_name google/flan-t5-base \
            --output_dir ${MODEL_DIR}/calibrate_${cali}_ibn_${ibn}_prompt_10 \
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
            --relevant_prompt "true true true true true true true true true true" \
            --irrelevant_prompt "false false false false false false false false false false" \
            --do_train \
            --sample_random true  \
            --sample_topk $k \
            --enable_similarity_loss $ibn \
            --tau $tau  \
            --enable_calibration $cali  \
            --calibration_margin_ngrams 1 2 \
            --gamma $gamma \
            --run_name prompt=1_batch=4_sample=top${k}_cali=${cali}_gamma=${gamma}_ibn=${ibn}
    done
done

# # generation
# mkdir -p /workspace/results/scifact/
# for folder in ${MODEL_DIR}/calibrate*prompt_1;do
#     name=${folder##*/}
#     for ckpt in 20000;do
#         python3 generate.py \
#             --corpus_jsonl ~/datasets/scifact/corpus.jsonl \
#             --model_name  ${folder}/checkpoint-${ckpt} \
#             --tokenizer_name google/flan-t5-base \
#             --output_jsonl results/scifact/${name}-${ckpt}.jsonl \
#             --device cuda \
#             --num_relevance_scores 10 \
#             --num_relevant_prompt 10 \
#             --batch_size 32 \
#             --max_length 512 \
#             --max_new_tokens 64 \
#             --activate_prompt_attention 1 \
#             --num_beams 1
#     done
# done
#
# # evaluation
# echo "this is calibrate.sh (start)" >> eval.log
# for file in results/scifact/*calibrate*prompt_1.jsonl;do
#     python3 evaluate.py \
#         --corpus_jsonl ~/datasets/scifact/corpus.jsonl \
#         --prediction $file \
#         --encoder_name /work/jhju/gte-base \
#         --regressor_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#         --reranker_name /work/jhju/monot5-3b-inpars-v2-scifact \
#         --device cuda \
#         --batch_size 2 >> eval.log
# done
# echo "this is calibrate.sh (end)" >> eval.log
