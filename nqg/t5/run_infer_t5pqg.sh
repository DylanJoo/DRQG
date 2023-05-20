DIR=evaluation/t5pqg/
mkdir -p $DIR

python3 inference_t5pqg.py \
    --model_name t5-base \
    --model_path t5pqg/checkpoint-10000 \
    --input_jsonl data/triples.eval.small.v0.sample.jsonl \
    --output_jsonl $DIR/triples.eval.small.v0.sample.pred.bs10.jsonl \
    --device cuda:2 \
    --batch_size 2 \
    --beam_size 10 \
    --max_q_length 10 \
    --max_p_length 256 \
