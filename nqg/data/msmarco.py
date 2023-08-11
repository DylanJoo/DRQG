import os
import json
import random
import collections
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import argparse

def query_centric_dataset(path):
    dataset = load_dataset(
            'csv', data_files=path, delimiter='\t', 
            column_names=['query', 'positive', 'negative']
    )
    print(f"Number of examples in dataset: {len(dataset['train'])}")
    return dataset

def passage_centric_dataset(path):
    print(f"Load data from: {path}...")
    dataset = load_dataset('json', data_files=path)
    print(f"Number of examples: {len(dataset['train'])}")
    return dataset

def convert_to_passage_centric(args):
    from data_utils import load_collection
    """ 
    Use msmarco officially released `train.triplet.small.tsv`.
    """
    triplet = collections.defaultdict(dict)
    collection = load_collection(args.collection, inverse=True)

    # load triplet with p-centered
    with open(args.input_tsv, 'r') as f:
        for line in tqdm(f):
            query, positive, negative = line.strip().split('\t')
            ## positive relevnace
            pid = collection[positive.strip()]
            if pid not in triplet:
                triplet[pid]['positive'] = set()
                triplet[pid]['negative'] = set()
            triplet[pid]['positive'].update([query])

            ## negative relevnace
            pid = collection[negative.strip()]
            if pid not in triplet:
                triplet[pid]['positive'] = set()
                triplet[pid]['negative'] = set()
            triplet[pid]['negative'].update([query])

    # calculate stats
    pos, neg = [], []
    for pid in triplet:
        pos.append(len(triplet[pid]['positive']))
        neg.append(len(triplet[pid]['negative']))

    with open(f"{args.output_jsonl.replace('json', 'stats.txt')}", 'w') as fstat:
        fstat.write(f"# passages: {len(triplet)}\n")
        fstat.write(f"# avg. positive (query): {np.mean(pos)}\n")
        fstat.write(f"# avg. negative (query): {np.mean(neg)}\\n")
        fstat.write(f"# passages has zero positive: {(np.array(pos) == 0).sum()}\n")
        fstat.write(f"# passages has zero negative: {(np.array(neg) == 0).sum()}\n")

    # reverse the mapping for output jsonl
    collection = {v: k for k, v in collection.items()}
    # output jsonl
    with open(args.output_jsonl, 'w') as f:
        for pid, queries in triplet.items():
            positives = list(queries['positive'])
            negatives = list(queries['negative'])
            n_pos = len(positives)
            n_neg = len(negatives)

            if n_pos >= args.min_n and n_neg >= args.min_n:
                f.write(json.dumps({
                    "passage": collection[pid],
                    "positive": positives, 
                    "negative": negatives
                }, ensure_ascii=False)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tsv", type=str, default=None)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument("--min_n", type=int, default=1, 
            help='the minumum number of obtained negative query.')
    args = parser.parse_args()
    
    convert_to_passage_centric(args)
