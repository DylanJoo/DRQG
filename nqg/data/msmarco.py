import os
import json
import random
import collections
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

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

    print(f"# passages: {len(triplet)}")
    print(f"# avg. positive (query): {np.mean(pos)}")
    print(f"# avg. negative (query): {np.mean(neg)}")
    print(f"# passages has zero positive: {(np.array(pos) == 0).sum()}")
    print(f"# passages has zero negative: {(np.array(neg) == 0).sum()}")

    # reverse the mapping for output jsonl
    collection = {v: k for k, v in collection.items()}
    # output jsonl
    with open(args.output_jsonl, 'w') as f:
        for pid, queries in triplet.items():
            positives = list(queries['positive'])
            negatives = list(queries['negative'])
            n_pos = len(positives)
            n_neg = len(negatives)

            # Setting1: neatives inner-join 
            # each p contains at most n_neg instances with same positive
            if 'v1' in args.output_jsonl:
                if (n_pos > 0) and (n_pos < n_neg):
                    positives = (positives * n_neg)[:n_neg]

            # Setting0: inner join (each p contains at most min(n_pos, n_neg))

            for pos, neg in zip(positives, negatives):
                f.write(json.dumps({
                    "passage": collection[pid],
                    "positive": pos, 
                    "negative": neg
                }, ensure_ascii=False)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tsv", type=str, default=None)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, default=None)
    args = parser.parse_args()
    
    convert_to_passage_centric(args)
