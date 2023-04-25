DIR=data/t5vqg_v0/
mkdir -p $DIR

python3 inference.py \
    --model_name t5-base \
    --model_path /home/jhju/models/t5vqg_v0/checkpoint-10000 \
    --input_jsonl data/triples.eval.small.v0.sample.jsonl \
    --output_jsonl $DIR/triples.eval.small.v0.sample.pred.bs10.jsonl \
    --decoding_type gaussian \
    --n_samples 10 \
    --flags positive negative \
    --device cuda:2 \
    --batch_size 4 \
    --beam_size 10 \
    --latent_size 256 \
    --max_length 10 \

    # --do_sample \
    # --top_k 10
