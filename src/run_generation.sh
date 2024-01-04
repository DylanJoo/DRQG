# trec-covid webis-touche2020 dbpedia-entity climate-fever scifact 

for dataset in arguana nfcorpus scidocs fiqa;do
    mkdir -p /workspace/results/${dataset}/
    for folder in models/checkpoint/*$1*;do
        name=${folder##*/}
        if [[ "$folder" != *"log" ]]; then
            for ckpt in 20000;do
                python3 generate.py \
                    --corpus_jsonl ~/datasets/${dataset}/corpus.jsonl \
                    --model_name  ${folder}/checkpoint-${ckpt} \
                    --tokenizer_name google/flan-t5-base \
                    --output_jsonl results/${dataset}/${name}-${ckpt}.jsonl \
                    --device cuda:0 \
                    --num_relevance_scores 10 \
                    --num_relevance_prompt 5 \
                    --batch_size 32 \
                    --max_length 512 \
                    --max_new_tokens 64 \
                    --activate_prompt_attention 1 \
                    --num_beams 1
            done
        fi
    done
done
