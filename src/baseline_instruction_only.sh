export CUDA_VISIBLE_DEVICES=0

TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl
EVAL_FILE=data/ce.minilm.hardneg.vL.eval.small.jsonl
MODEL_DIR=/work/jhju/readqg-baseline

for bs in 4;do
    let m=32/2/$bs
    python3 train/train_baseline.py \
        --model_name_or_path google/flan-t5-base \
        --tokenizer_name google/flan-t5-base \
        --config_name google/flan-t5-base \
        --output_dir ${MODEL_DIR}/baseline_instruct_with_num \
        --max_p_length 128 \
        --max_q_length 16 \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --m_positive_per_example $m \
        --m_negative_per_example $m \
        --learning_rate 1e-2 \
        --lr_scheduler_type constant \
        --max_steps 20000 \
        --save_steps 10000 \
        --train_file ${TRAIN_FILE} \
        --instruction_prompt "Generate a question for the passage with relevance label: " \
        --baseline_prefix "{0} passage: {1}" \
        --do_train \
        --run_name ReadQG-baseline-prefix > ${MODEL_DIR}/baseline_instruct_with_num.log
done

# # generation
# mkdir -p /workspace/results/scifact/
# for folder in ${MODEL_DIR}/baseline_instruct_with_num*;do
#     name=${folder##*/}
#     for ckpt in 20000;do
#         python3 generate.py \
#             --corpus_jsonl ~/datasets/scifact/corpus.jsonl \
#             --model_name  ${folder}/checkpoint-${ckpt} \
#             --tokenizer_name google/flan-t5-base \
#             --output_jsonl results/scifact/${name}-${ckpt}.jsonl \
#             --device cuda:0 \
#             --num_relevance_scores 10 \
#             --num_relevance_prompt 5 \
#             --batch_size 32 \
#             --max_length 512 \
#             --max_new_tokens 64 \
#             --activate_prompt_attention 1 \
#             --num_beams 1
#     done
# done
#
# # evaluation
# for file in results/scifact_greedy/baseline_instruct_with_num.jsonl;do
#     python3 evaluate.py \
#         --corpus_jsonl ~/datasets/scifact/corpus.jsonl \
#         --prediction $file \
#         --encoder_name /work/jhju/gte-base \
#         --regressor_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#         --reranker_name /work/jhju/monot5-3b-inpars-v2-scifact \
#         --device cuda:0 \
#         --batch_size 2 >> eval.log
# done
