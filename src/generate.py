import json
import copy
import torch
import argparse
import collections
import numpy as np
from tqdm import tqdm
# model
from models import FlanT5
from transformers import AutoConfig, AutoTokenizer
# data
from data import DataCollatorBase
from datasets import load_dataset
from torch.utils.data import DataLoader
from arguments_new import *
from utils import batch_iterator

def get_model_class(name):
    if 'flan' in name:
        return FlanT5 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # load model
    parser.add_argument("--model_name")
    parser.add_argument("--model_path")
    parser.add_argument("--input_jsonl")
    parser.add_argument("--output_jsonl")
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--batch_size", default=2, type=int)

    # generation config
    parser.add_argument("--num_pred", default=6, type=int)
    parser.add_argument("--prefix", default=None, type=str)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--max_q_length", default=512, type=int)
    parser.add_argument("--max_p_length", default=64, type=int)
    parser.add_argument("--do_sample", default=False, action='store_true')
    parser.add_argument("--top_k", default=10, type=int)

    args = parser.parse_args()

    # Model
    ## config and tokenizers
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    used_scores = list(range(0, 101, 101//(args.num_pred-1)))
    used_scores = [s*0.01 for s in used_scores]

    ## checkpoints
    model_class = get_model_class(args.model_path)
    model = model_class.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()

    # Data
    ## datacollator
    data_collator = DataCollatorBase(
            tokenizer=tokenizer, 
            max_p_length=args.max_p_length,
            device=args.device,
            scores=used_scores,
            prefix=args.prefix
    )

    # Data
    ## dataset, dataloader
    dataset = load_dataset("json", data_files=args.input_jsonl)['train']

    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            pin_memory=False,
            shuffle=False,
            collate_fn=data_collator
    )

    # Generation
    with torch.no_grad(), open(args.output_jsonl, 'w') as fout:
        for batch_inputs, batch_texts in tqdm(dataloader):
            batch_inputs = batch_inputs.to(args.device)
            output_ids = model.generate(
                    **batch_inputs,
                    num_beams=args.num_beams,
                    max_length=args.max_q_length,
                    do_sample=args.do_sample,
                    top_k=args.top_k,
            )

            output_texts = tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True
            )

            # decode
            for i, s_and_e in enumerate(
                    batch_iterator(output_texts, args.num_pred, True)
                ):
                s, e = s_and_e
                fout.write(json.dumps({
                    "passage": batch_texts[i],
                    "condition_score": used_scores,
                    "generated_query": output_texts[s: e]
                }) + '\n')

    print('done')


