touch eval.log
RESULT_DIR=/work/jhju/beir-readqg
decoding=top10

echo greedy >> eval.log

FILE=${RESULT_DIR}/scifact_${decoding}/calibrate_rank_ibn_dd-20000.jsonl
python3 evaluate.py \
    --corpus_jsonl ~/datasets/scifact/corpus.jsonl \
    --prediction $FILE \
    --encoder_name thenlper/gte-base \
    --reranker_name zeta-alpha-ai/monot5-3b-inpars-v2-scifact \
    --device cuda:0 \
    --batch_size 2 >> eval.log

