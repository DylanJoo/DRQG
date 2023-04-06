export CUDA_VISIBLE_DEVICES=0
python3 inference.py \
    --model_name t5-base \
    --model_path /home/jhju/models/t5vqg_v0/checkpoint-10000 \
    --input_jsonl data/triples.train.small.v0.sample.jsonl \
    --output_jsonl data/triples.train.small.v0.sample.pred.jsonl \
    --positive \
    --device cuda:0 \
    --batch_size 8 \
    --latent_size 256 \
    --beam_size 5 \
    --max_length 20
