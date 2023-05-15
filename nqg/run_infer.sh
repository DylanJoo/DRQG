for MODEL in bartvqgspt/*;do
    EVAL_DATA=/home/jhju/datasets/msmarco.triples_train_small/triples.train.small.v0.sample.jsonl
    DIR=evaluation/$MODEL
    mkdir -p $DIR
    python3 inference.py \
        --model_name facebook/bart-base \
        --model_path $MODEL/checkpoint-5000 \
        --input_jsonl $EVAL_DATA \
        --output_jsonl $DIR/triples.eval.small.v0.sample.pred.jsonl \
        --generation_type gaussian \
        --flags prediction  \
        --device cuda:1 \
        --batch_size 1 \
        --latent_size 128 \
        --max_q_length 10 \
        --max_p_length 128 \
        --n_soft_prompts 1 \
        --n_side_tail 5 
done
