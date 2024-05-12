export CUDA_VISIBLE_DEVICES=0

MODEL_DIR=/work/jhju/readqg-baseline
RESULT_DIR=/work/jhju/readqg-results
# trec-covid webis-touche2020 dbpedia-entity climate-fever scifact 

for dataset in scifact arguana fiqa nfcorpus scidocs;do
# for dataset in scifact;do
    decoding=greedy
    # decoding=_top10
    # decoding=_beam3
    mkdir -p ${RESULT_DIR}/${dataset}_${decoding}/

    for folder in ${MODEL_DIR}/*$1*;do
        name=${folder##*/}
        if [[ "$folder" != *"log" ]]; then
            echo ${folder}
            for ckpt in 20000;do
                python3 generate.py \
                    --corpus_jsonl ~/datasets/${dataset}/corpus.jsonl \
                    --model_name  ${folder}/checkpoint-${ckpt} \
                    --tokenizer_name google/flan-t5-base \
                    --output_jsonl ${RESULT_DIR}/${dataset}_${decoding}/${name}-${ckpt}.jsonl \
                    --device cuda \
                    --num_relevance_scores 10 \
                    --num_relevant_prompt 1 \
                    --batch_size 32 \
                    --max_length 512 \
                    --max_new_tokens 64 \
                    --activate_prompt_attention 1 \
                    --do_sample 
            done
        fi
    done

done
