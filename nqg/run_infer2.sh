model=t5vqgv0
DIR=evaluation/$model
mkdir -p $DIR

python3 inference2.py \
    --model_name t5-base \
    --model_path $model/checkpoint-5000 \
    --input_jsonl data/triples.eval.small.v0.sample.jsonl \
    --output_jsonl $DIR/triples.eval.small.v0.sample.pred.bs10.jsonl \
    --generation_type gaussian \
    --flags positive negative \
    --device cuda:2 \
    --batch_size 2 \
    --latent_size 128 \
    --max_q_length 10 \
    --max_p_length 128 \
    --n_samples 2 \
    --beam_size 3

