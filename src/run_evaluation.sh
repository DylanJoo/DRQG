reranker_hf=/work/jhju/monot5-3b-inpars-v2-scifact

touch eval.log
for file in results/*$1*.jsonl;do
    name=${file##*/}
    python3 evaluate.py \
        --corpus_jsonl ~/datasets/scifact/corpus.jsonl \
        --prediction $file \
        --encoder_name DylanJHJ/gtr-t5-base \
        --regressor_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --reranker_name $reranker_hf \
        --device cuda:0 \
        --batch_size 2 >> eval.1.log
done

# touch eval.msmarco.log
# for file in results/msmarco/*.jsonl;do
#     name=${file##*/}
#     python3 evaluate.py \
#         --corpus_jsonl ~/datasets/scifact/corpus.jsonl \
#         --prediction $file \
#         --encoder_name DylanJHJ/gtr-t5-base \
#         --ranker_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#         --device cuda:1 \
#         --batch_size 2 >> eval.msmarco.log
# done
