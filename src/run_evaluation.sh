export CUDA_VISIBLE_DEVICES=0

touch eval.log

RESULT_DIR=/work/jhju/readqg-results

echo greedy >> eval.log
for file in ${RESULT_DIR}/scifact_greedy/*$1*.jsonl;do
    python3 evaluate.py \
        --corpus_jsonl ~/datasets/scifact/corpus.jsonl \
        --prediction $file \
        --encoder_name /work/jhju/gte-base \
        --reranker_name /work/jhju/monot5-3b-inpars-v2-scifact \
        --device cuda:0 \
        --batch_size 2 >> eval.log
done


# echo baseline >> eval.log
# for file in results/scifact_greedy/*num*.jsonl;do
#     python3 evaluate.py \
#         --corpus_jsonl ~/datasets/scifact/corpus.jsonl \
#         --prediction $file \
#         --reranker_name /work/jhju/monot5-3b-inpars-v2-scifact \
#         --device cuda:0 \
#         --batch_size 2 >> eval.log
# done
