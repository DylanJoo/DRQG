export CUDA_VISIBLE_DEVICES=2
BASE=bartvqg
MODEL=nils

rm -rvf $BASE/$MODEL
PRT_MODEL=bartqg-d2q/relevant/checkpoint-8000
PRT_CONFIG=facebook/bart-base

TRAIN_FILE=/home/jhju/datasets/nils.sentence.transformers/ce.minilm.hardneg.vL.jsonl
python3 train_vqg.py \
  --model_name_or_path $PRT_MODEL \
  --tokenizer_name $PRT_CONFIG \
  --config_name $PRT_CONFIG \
  --output_dir $BASE/$MODEL \
  --max_p_length 200 \
  --max_q_length 16 \
  --per_device_train_batch_size 4 \
  --m_negative_per_example 4 \
  --m_positive_per_example 4 \
  --learning_rate 1e-3 \
  --evaluation_strategy steps \
  --max_steps 10000 \
  --save_steps 2000 \
  --eval_steps 500 \
  --train_file $TRAIN_FILE \
  --latent_size 128 \
  --has_compressed_layer true \
  --add_classification_head true \
  --annealing_fn cyclic \
  --n_cycle 10 \
  --pos_anchors 'what where when who how why which' \
  --neg_anchors 'what where when who how why which' \
  --pooling mean \
  --activation tanh \
  --warmup_ratio 0.1 \
  --do_train \
  --do_eval 

# [NOTE] 
# larger leanring rate 
# longer and diversified anchors  
# n cycle  
# larger negative samples  
