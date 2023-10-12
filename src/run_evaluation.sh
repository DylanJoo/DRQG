for file in results/*.jsonl;do
    name=${file##*/}
    echo $name >> eval.log

    python3 evaluate.py \
        --corpus_jsonl ~/datasets/scifact/corpus.jsonl \
        --prediction $file \
        --encoder_name DylanJHJ/gtr-t5-base \
        --ranker_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --device cuda:0 \
        --batch_size 2 >> eval.log
done
