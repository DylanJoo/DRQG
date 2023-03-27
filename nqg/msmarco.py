import os
import collections
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

def load_collection(path):
    collection = {}
    with open(path, 'r') as f:
        for line in tqdm(f):
            pid, content = line.strip().split('\t')
            collection[pid] = content
    print("load collection done")
    return collection

def load_queries(path):
    queries = {}
    with open(path, 'r') as f:
        for line in tqdm(f):
            qid, content = line.strip().split('\t')
            if int(rel) == 1:
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
            if query not in triplet:
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

def load_triplet_ids(path, stats=False):
    triplet = collections.defaultdict(dict)
    with open(path, 'r') as f:
        for line in tqdm(f):
            qid, pos_pid, neg_pid = line.strip().split('\t')
            if 'positive' not in triplet[qid]:
                triplet[qid]['positive'] = set()
                triplet[qid]['negative'] = set()
            triplet[qid]['positive'].update(pos_pid)
            triplet[qid]['negative'].update(neg_pid)
    print("load triplet ids done")

    if stats:
        pos, neg = [], []
        for key in triplet:
            pos.append(len(triplet[key]['positive']))
            neg.append(len(triplet[key]['negative']))

        print(f"# Number of queries: {len(triplet)}")
        print(f"# Number of average positive: {np.mean(pos)}")
        print(f"# Number of average negative: {np.mean(neg)}")
    return triplet
            
## (1) doc2query dataset 
# i.e., query generation task
def doc2query_dataset(args):
    dataset = None
    collection = load_collection(args.collection)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    return dataset

## (2) Triplet dataset
def triplet_dataset(args):
    dataset = load_dataset('csv', 
            data_files=args.triplet, 
            delimiter='\t',
            column_names=['query', 'positive', 'negative'],
    )
    print(f"Number of instances in dataset: {len(dataset['train'])}")
    return dataset
