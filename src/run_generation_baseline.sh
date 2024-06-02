MODEL_DIR=/work/jhju/flan-t5-readqg.msmarco-harg-neg
RESULT_DIR=/work/jhju/beir-readqg
NUM_PROMPT=5

# for dataset in scifact arguana fiqa nfcorpus scidocs;do
for dataset in scifact;do
    # decoding=greedy
    # decoding=beam3
    decoding=top10
    mkdir -p ${RESULT_DIR}/${dataset}${decoding}/

    for folder in ${MODEL_DIR}/*$1*;do
        name=${folder##*/}
        echo ${folder}
        for ckpt in 20000;do
            python3 generate.py \
                --corpus_jsonl ~/datasets/${dataset}/corpus.jsonl \
                --model_name  ${folder}/checkpoint-${ckpt} \
                --tokenizer_name google/flan-t5-base \
                --output_jsonl ${RESULT_DIR}/${dataset}_${decoding}/${name}-${ckpt}.jsonl \
                --device cuda \
                --num_relevance_scores 10 \
                --prefix '{0} passage: {1}' \
                --batch_size 32 \
                --max_length 512 \
                --max_new_tokens 64 \
                --activate_prompt_attention 1 \
                --do_sample \
                --top_k 10
        done
    done
done
