DIR=evaluation/t5vqg_v0/
mkdir -p $DIR

python3 inference.py \
    --model_name t5-base \
    --model_path t5vqg_v0/checkpoint-10000 \
    --input_jsonl data/triples.eval.small.v0.sample.jsonl \
    --output_jsonl $DIR/triples.eval.small.v0.sample.pred.top10.jsonl \
    --decoding_type gaussian \
    --flags positive negative \
    --device cuda:2 \
    --batch_size 2 \
    --latent_size 256 \
    --max_q_length 10 \
    --max_p_length 256 \
    --do_sample \
    --top_k 10
    # --beam_size 10 \
