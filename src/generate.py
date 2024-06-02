import json
import copy
import torch
import argparse
import collections
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils import batch_iterator
from transformers import AutoConfig, AutoTokenizer
from evaluation import READGen
from datasets import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_jsonl")
    # generator
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--tokenizer_name")
    parser.add_argument("--output_jsonl")
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--num_relevance_scores", type=int)
    parser.add_argument("--activate_prompt_attention", type=int, default=1)
    parser.add_argument("--prefix", type=str, default=None)
    # generation config
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--max_new_tokens", default=64, type=int)
    parser.add_argument("--do_sample", default=False, action='store_true')
    parser.add_argument("--top_k", default=None, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--num_relevant_prompt", default=0, type=int)

    args = parser.parse_args()

    # READ Generator
    generator = READGen(
            model_path=args.model_name_or_path, 
            tokenizer_name=args.tokenizer_name,
            relevance_scores=None,
            num_relevance_scores=args.num_relevance_scores,
            output_jsonl=args.output_jsonl,
            device=args.device,
            num_relevant_prompt_idx=args.num_relevant_prompt
    )
    generator.model.to(args.device)
    generator.model.eval()

    # READ kwargs
    generator.model.read_kwargs = {
            'activate_prompt_attention': bool(args.activate_prompt_attention)
    }

    # Data
    ## [NOTE] preprocess here.
    data = []
    with open(args.corpus_jsonl, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())

            # customize each datasets here.
            # msmarco
            if 'msmarco' in args.corpus_jsonl:
                data.append({'doc_id': '', 'passage': item['passage']})
            else:
                docid = item.get('_id')
                title = item.get('title', "")
                text = item.get('text', "")
                content = title + " " + text
                data.append({'doc_id': docid, 'passage': content})

    dataset = Dataset.from_list(data)
    print(dataset)

    # Generation
    data_iterator = batch_iterator(dataset, args.batch_size, False)
    with torch.no_grad(), open(args.output_jsonl, 'w') as fout:
        for batch in tqdm(data_iterator, total=len(dataset)//args.batch_size+1):
            outputs = generator.batch_generate(
                    batch['passage'],
                    max_length=args.max_length,
                    prefix=args.prefix,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=args.do_sample,
                    top_k=args.top_k,
                    top_p=args.top_p,
            )

            # writer
            for i, docid in enumerate(batch['doc_id']):
                fout.write(json.dumps({
                    "doc_id": docid,
                    "relevance_scores": generator.relevance_scores.cpu().tolist(),
                    "generated_query": outputs[i]
                }) + '\n')

    print('done')

