touch eval.log

for file in results/scifact/*$1*.jsonl;do
    python3 evaluate.py \
        --corpus_jsonl ~/datasets/scifact/corpus.jsonl \
        --prediction $file \
        --encoder_name DylanJHJ/gtr-t5-base \
        --regressor_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --reranker_name /work/jhju/monot5-3b-inpars-v2-scifact \
        --device cuda:0 \
        --batch_size 2 >> eval.log
done
