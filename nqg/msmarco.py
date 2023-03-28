import os
import json
import random
import collections
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

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

## (3) Triplet dataset
def query_aligned_triplet_dataset(args):
    if not os.path.exists(args.q_aligned_triplet):
        ## Step1: load triplet dictionary
        triplet = load_triplet(args.triplet, True)
        ## Step2: convert to jsonl and saved
        convert_triplet_to_jsonl(triplet, args.q_aligned_triplet)

    dataset = load_dataset('json', data_files=args.q_aligned_triplet)
    print(f"Number of instances in dataset: {len(dataset['train'])}")
    return dataset

## (4) Triplet dataset with passage aligned
def passage_aligned_triplet_dataset(args):
    # if not os.path.exists(args.p_aligned_triplet):
    if True:
        triplet = collections.defaultdict(dict)
        collection = load_collection(args.collection, inverse=True)

        # load triplet with p aligned
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
        with open(args.p_aligned_triplet, 'w') as f:
            for pid, queries in triplet.items():
                # Setting1: inner join
                for pos, neg in zip(queries['positive'], 
                                    queries['negative']):
                    f.write(json.dumps({
                        "passage": collection[pid],
                        "positive": pos, 
                        "negative": neg
                    }, ensure_ascii=False)+'\n')
    else:
        print(f"Load data from: {args.p_aligned_triplet}...")
    dataset = load_dataset('json', data_files=args.p_aligned_triplet)
    print(f"Number of instances: {len(dataset['train'])}")
    return dataset

# def load_triplet(path, stats=False):
#     triplet = collections.defaultdict(dict)
#     with open(path, 'r') as f:
#         for line in tqdm(f):
#             query, positive, negative = line.strip().split('\t')
#             if query not in data['query']:
#                 triplet[query]['positive'] = set()
#                 triplet[query]['negative'] = set()
#             triplet[query]['negative'].update([negative])
#             triplet[query]['positive'].update([positive])
#     print("load triplet done")
#
#     if stats:
#         pos, neg = [], []
#         for key in triplet:
#             pos.append(len(triplet[key]['positive']))
#             neg.append(len(triplet[key]['negative']))
#
#         print(f"# Number of queries: {len(triplet)}")
#         print(f"# Number of average positive: {np.mean(pos)}")
#         print(f"# Number of average negative: {np.mean(neg)}")
#     return triplet

def convert_triplet_to_jsonl(triplet, output_path):
    with open(output_path, 'w') as f:
        for query, passages in tqdm(triplet.items()):
            positives = list(passages['positive'])
            negatives = list(passages['negative'])
            N = len(negatives)

            f.write(json.dumps({
                "query": query.strip(), 
                "positive": random.choices(positives, k=N),
                "negative": random.sample(negatives, k=N)
            }, ensure_ascii=False)+'\n')
    return 0


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
