import os
import json
import random
import collections
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

## (1) doc2query dataset 
# i.e., query generation task
# [TODO] this code is unfinished
def doc2query_dataset(args):
    dataset = None
    collection = load_collection(args.collection)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    return dataset

## (2) Triplet dataset (query-centric)
def triplet_dataset(args):
    dataset = load_dataset('csv', 
            data_files=args.triplet, 
            delimiter='\t',
            column_names=['query', 'positive', 'negative'],
    )
    print(f"Number of instances in dataset: {len(dataset['train'])}")
    return dataset

## (4) Triplet dataset (passage-centric)
def passage_centric_triplet_dataset(args):
    path = args.train_file or args.p_centric_triplet
    if not os.path.exists(path):
        triplet = collections.defaultdict(dict)
        collection = load_collection(args.collection, inverse=True)

        # load triplet with p-centered
        with open(args.triplet, 'r') as f:
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
        for key in triplet:
            pos.append(len(triplet[key]['positive']))
            neg.append(len(triplet[key]['negative']))
        print(f"# passages: {len(triplet)}")
        print(f"# avg. positive (query): {np.mean(pos)}")
        print(f"# avg. negative (query): {np.mean(neg)}")
        print(f"# passages has zero positive: {(np.array(pos) == 0).sum()}")
        print(f"# passages has zero negative: {(np.array(neg) == 0).sum()}")

        # output jsonl
        collection = load_collection(args.collection)
        with open(args.path, 'w') as f:
            for pid, queries in triplet.items():
                # Setting0: inner join with all negative. For each p
                ## Contains at most min(n_pos, n_neg) instances
                positives = list(queries['positive'])
                negatives = list(queries['negative'])
                n_pos = len(positives)
                n_neg = len(negatives)

                # Setting1: neatives innger-join. 
                ## each p contains at most n_neg*2 instances
                if args.joinbynegative:
                    if (n_pos > 0) and (n_pos < n_neg):
                        positives = (positives * n_neg)[:n_neg]

                for pos, neg in zip(positives, negatives):
                    f.write(json.dumps({
                        "passage": collection[pid],
                        "positive": pos, 
                        "negative": neg
                    }, ensure_ascii=False)+'\n')

    else:
        print(f"Load data from: {path}...")
    dataset = load_dataset('json', data_files=path)
    print(f"Number of instances: {len(dataset['train'])}")
    return dataset


### Load entities
def load_collection(path, inverse=False):
    collection = {}
    with open(path, 'r') as f:
        for line in tqdm(f):
            pid, content = line.strip().split('\t')
            if inverse:
                collection[content.strip()] = pid
            else:
                collection[pid] = content.strip()
    print("load collection done")
    return collection

def load_queries(path, inverse=False):
    queries = {}
    with open(path, 'r') as f:
        for line in tqdm(f):
            qid, content = line.strip().split('\t')
            if inverse:
                queries[content] = qid
            else:
                queries[qid] = content
    print("load queries done")
    return qrels

def load_qrels(path):
    qrels = {}
    with open(path, 'r') as f:
        for line in tqdm(f):
            qid, _, pid, rel = line.strip().split('\t')
            if int(rel) == 1:
                qrels[qid].append(pid)
    print("load qrels done")
    return qrels

def load_triplet(path, stats=False):
    triplet = collections.defaultdict(dict)
    with open(path, 'r') as f:
        for line in tqdm(f):
            query, positive, negative = line.strip().split('\t')
            if query not in data['query']:
                triplet[query]['positive'] = set()
                triplet[query]['negative'] = set()
            triplet[query]['negative'].update([negative])
            triplet[query]['positive'].update([positive])
    print("load triplet done")

    if stats:
        pos, neg = [], []
        for key in triplet:
            pos.append(len(triplet[key]['positive']))
            neg.append(len(triplet[key]['negative']))

        print(f"# Number of queries: {len(triplet)}")
        print(f"# Number of average positive: {np.mean(pos)}")
        print(f"# Number of average negative: {np.mean(neg)}")
    return triplet
