MODEL_DIR=/work/jhju/flan-t5-readqg.msmarco-harg-neg
RESULT_DIR=/work/jhju/beir-readqg
NUM_PROMPT=5

# for dataset in scifact arguana fiqa nfcorpus scidocs;do
for dataset in scifact;do
    decoding=top10
    mkdir -p ${RESULT_DIR}/${dataset}_${decoding}/

    for folder in ${MODEL_DIR}/calibrate_rank_ibn_dd;do
        name=${folder##*/}
        echo ${folder}
        for ckpt in 20000;do
            python3 generate.py \
                --corpus_jsonl ~/datasets/${dataset}/corpus.jsonl \
                --model_name  ${folder}/checkpoint-${ckpt} \
                --tokenizer_name google/flan-t5-base \
                --output_jsonl ${RESULT_DIR}/${dataset}_${decoding}/${name}-${ckpt}.jsonl \
                --device cuda:0 \
                --num_relevance_scores 10 \
                --num_relevant_prompt ${NUM_PROMPT} \
                --batch_size 32 \
                --max_length 384 \
                --max_new_tokens 64 \
                --activate_prompt_attention 1 \
                --do_sample \
                --top_p 10
        done
    done
done
