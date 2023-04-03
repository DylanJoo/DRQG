import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from models import T5VQG
from data import DataCollatorForT5VQG
from torch.utils.data import DataLoader

def generate(inputs, max_length, device, model, tokenizer, f):

    passages = batch.pop('passage')
    positives = batch.pop('positive')
    negatives = batch.pop('negative')

    for k in inputs:
        inputs[k] = inputs[k].cuda(device)

    # write outputs for positive
    positives_pred = []
    outputs = model.generate(**inputs, max_length=max_length)
    for o in outputs:
        positives_pred.append(tokenizer.decode(o, skip_special_tokens=True))

    negatives_pred = []
    outputs = model.generate(**inputs, max_length=max_length)
    for o in outputs:
        negatives_pred.append(tokenizer.decode(o, skip_special_tokens=True))

    # wrap two outputs
    for i in range(len(passages)):
        f.write(json.dumps({
            "passage": passages[i],
            "positive_prediction": positives_pred[i], 
            "positive": positives[i],
            "negative_prediction": negatives_pred[i], 
            "negative": negatives[i]
        }, ensure_ascii=False)+'\n')
        # Evaluation codes if existed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--used_checkpoint", type=str, default=None)
    parser.add_argument("--used_tokenizer", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    # load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.used_tokenizer)
    dataset = load_dataset('json', data_files=args.jsonl_file)['train']

    model = T5VQG.from_pretrained(args.used_checkpoint).to(args.device)
    # [TODO]
    # The model should do something different to make it generate postive/negative query with variational inferfenece

    # organize dataset/dataloader
    datacollator = DataCollatorForT5VQG(
            tokenizer=tokenizer,
            padding=True,
            return_tensors='pt',
            is_train=False,
            max_length=args.max_length,
    )
    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=False,
            collate_fn=datacollator
    )

    # predict and write
    with torch.no_grad(), open(args.output_file, 'w') as f:
        for batch in tqdm(dataloader):

            generate(inputs=batch, 
                     max_length=args.max_length,
                     device=args.device,
                     model=model, 
                     tokenizer=tokenizer,
                     f=f
