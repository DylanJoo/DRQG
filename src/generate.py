import json
import copy
import torch
import argparse
import collections
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from arguments import *
from utils import batch_iterator
from transformers import AutoConfig, AutoTokenizer

# model
from models import FlanT5, SoftPromptFlanT5
def get_model_class(path):
    if 'prefix' in path:
        model_class = FlanT5 
    if 'soft' in path:
        model_class = SoftPromptFlanT5
    else:
        model_class = Flan
    return model_class

# data
from data import DataCollatorForBaseline, DataCollatorForPromptQG
def get_data_collator_clss(path):
    if 'prefix' in path:
        data_class = DataCollatorForBaseline
    if 'soft' in path:
        data_class = DataCollatorForPromptQG
    else:
        data_class = DataCollatorForBaseline
    return data_class
## get prompts
def convert_string_to_idx(string):
    if string is not None:
        tokenized_idices = tokenizer.encode(string, add_special_tokens=False)
        return [tokenized_idices, len(tokenized_idices)]
    else:
        return [None, 0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # load model
    parser.add_argument("--model_name")
    parser.add_argument("--model_path")
    parser.add_argument("--input_jsonl")
    parser.add_argument("--output_jsonl")
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--instruction_prompt", default=None, type=str)
    parser.add_argument("--relevance_prompt", default=None, type=str)

    # generation config
    parser.add_argument("--num_pred", default=6, type=int)
    parser.add_argument("--prefix", default='{1}', type=str)
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
    ### mapping the string to prompts
    instruction_prompt = convert_string_to_idx(args.instruction_prompt)
    relevance_prompt = convert_string_to_idx(args.relevance_prompt)
    relevance_prompt[1] = min(1, relevance_prompt[1])
    model = get_model_class(args.model_path).from_pretrained(
            args.model_path, 
            instruction_prompt[0], 
            relevance_prompt[0]
    )
    model.to(args.device)
    model.eval()

    # Data
    ## datacollator
    data_collator = get_data_collator_clss(args.model_path)(
            tokenizer=tokenizer, 
            max_p_length=args.max_p_length,
            device=args.device,
            scores=used_scores,
            prefix=args.prefix,
            prompt_length=instruction_prompt[1] + relevance_prompt[1]
    )
    collate_fn = lambda ft: data_collator(ft, is_eval=True)

    # Data
    ## dataset, dataloader
    dataset = load_dataset("json", data_files=args.input_jsonl)['train']

    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            pin_memory=False,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True
    )

    # Generation
    if args.relevance_prompt is not None:
        rel_scores = torch.Tensor(used_scores).to(model.device)
        rel_scores = rel_scores.repeat(args.batch_size)
    else:
        rel_scores = None

    with torch.no_grad(), open(args.output_jsonl, 'w') as fout:
        for batch_inputs, batch_texts in tqdm(dataloader):
            batch_inputs = batch_inputs.to(args.device)
            output_ids = model.generate(
                    **batch_inputs,
                    rel_scores=rel_scores,
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


