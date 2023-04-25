DIR=data/t5pqg/
mkdir -p $DIR

python3 inference_t5pqg.py \
    --model_name t5-base \
    --model_path /home/jhju/models/t5pqg/checkpoint-10000 \
    --input_jsonl data/triples.eval.small.v0.sample.jsonl \
    --output_jsonl $DIR/triples.eval.small.v0.sample.pred.top10.jsonl \
    --decoding_type interpolate_none \
    --n_samples 10 \
    --flags positive negative \
    --device cuda:2 \
    --batch_size 4 \
    --beam_size 5 \
    --latent_size 256 \
    --max_length 10 \
    --do_sample \
    --top_k 10
