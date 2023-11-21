# scidocs
for folder in models/checkpoint/*$1*;do
    name=${folder##*/}
    python3 generate.py \
        --corpus_jsonl ~/datasets/scifact/corpus.jsonl \
        --model_name models/checkpoint/${name}/checkpoint-20000 \
        --tokenizer_name google/flan-t5-base \
        --output_jsonl results/${name}.jsonl \
        --device cuda:0 \
        --num_relevance_scores 10 \
        --num_relevance_prompt 1 \
        --batch_size 2 \
        --max_length 512 \
        --max_new_tokens 64 \
        --num_beams 1
done

# mkdir -p results/msmarco/
# # msmarco
# for folder in models/drqg/flan*;do
#     name=${folder##*/}
#     python3 generate.py \
#         --corpus_jsonl ~/datasets/nils.sentence.transformers/msmarco.dev.jsonl \
#         --model_name models/drqg/${name}/checkpoint-20000 \
#         --tokenizer_name google/flan-t5-base \
#         --output_jsonl results/msmarco/${name}.jsonl \
#         --device cuda:0 \
#         --num_relevance_scores 10 \
#         --batch_size 2 \
#         --max_length 512 \
#         --max_new_tokens 64 \
#         --num_beams 1
# done
