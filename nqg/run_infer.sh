MODEL=t5vqgspt/BM25-0_P-20_Z-128_BS-4
DIR=evaluation/$MODEL
mkdir -p $DIR

python3 inference.py \
    --model_name t5-base \
    --model_path $MODEL/checkpoint-10000 \
    --input_jsonl data/triples.eval.small.v0.sample.jsonl \
    --output_jsonl $DIR/triples.eval.small.v0.sample.pred.bs10.jsonl \
    --generation_type gaussian \
    --flags prediction  \
    --device cuda:1 \
    --batch_size 1 \
    --latent_size 128 \
    --max_q_length 10 \
    --max_p_length 128 \
    --n_soft_prompts 20 \
    --n_samples 2 \
    --do_sample \
    --top_k 10
