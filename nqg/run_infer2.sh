model=t5vqg_v1
DIR=evaluation/$model
mkdir -p $DIR

python3 inference_dev.py \
    --model_name t5-base \
    --model_path $model/checkpoint-10000 \
    --input_jsonl data/triples.eval.small.v0.sample.jsonl \
    --output_jsonl $DIR/triples.eval.small.v0.sample.pred.top10.jsonl \
    --sampling gaussian \
    --device cuda:2 \
    --batch_size 2 \
    --latent_size 256 \
    --max_q_length 10 \
    --max_p_length 256 \
    --debug 0 \
    --top_k 10  --do_sample
    # --beam_size 10
