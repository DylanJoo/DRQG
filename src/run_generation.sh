for folder in models/drqg/flan*;do
    name=${folder##*/}
    python3 generate.py \
        --corpus_jsonl ~/datasets/scifact/corpus.jsonl \
        --model_name models/drqg/${name}/checkpoint-20000 \
        --tokenizer_name google/flan-t5-base \
        --output_jsonl results/${name}.jsonl \
        --device cuda:2 \
        --num_relevance_scores 10 \
        --batch_size 2 \
        --max_length 512 \
        --max_new_tokens 64 \
        --num_beams 1
done
