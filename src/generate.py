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
from evaluation import READGen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_jsonl")
    # generator
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--tokenizer_name")
    parser.add_argument("--output_jsonl")
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--num_relevance_scores", type=int)
    # generation config
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--max_new_tokens", default=64, type=int)
    parser.add_argument("--do_sample", default=False, action='store_true')
    parser.add_argument("--top_k", default=10, type=int)

    args = parser.parse_args()

    # READ Generator
    generator = READGen(
            model_path=args.model_name_or_path, 
            tokenizer_name=args.tokenizer_name,
            relevance_scores=None,
            num_relevance_scores=args.num_relevance_scores,
            output_jsonl=args.output_jsonl
    )
    generator.to(args.device)
    generator.eval()

    # Data
    ## [NOTE] preprocess here.
    data = []
    with open(args.corpus_jsonl, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())

            if isinstance(item['abstract'], list):
                content = " ".join(item['abstract'])
            else:
                content = item['abstract']

            data.append(
                    {'doc_id': item['doc_id'], 'passage': content}
            )

    dataset = Dataset.from_list(data)
    print(dataset)

    # Generation
    data_iterator = batch_iterator(dataset, args.batch_size, False)
    with torch.no_grad(), open(args.output_jsonl, 'w') as fout:
        for batch in tqdm(data_iterator, total=len(dataset).args.batch_size+1):
            outputs = generator.batch_generate(
                    batch['passage'],
                    max_length=512,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=args.do_sample,
                    top_k=args.top_k,
            )

            # writer
            for i, docid in enumerate(batch['doc_id']):
                fout.write(json.dumps({
                    "doc_id": docid,
                    "relevance_scores": generator.relevance_scores,
                    "generated_query": outputs[i]
                }) + '\n')

    print('done')

