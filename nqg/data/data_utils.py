import os
import json
import random
import collections
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

def load_collection(path, inverse=False):
    collection = {}
    with open(path, 'r') as f:
        for line in tqdm(f):
            try: # msmarco's collection
                pid, content = line.strip().split('\t')
            except:
                pid, content, title = line.strip().split('\t')

            if inverse:
                collection[content.strip()] = pid
            else:
                collection[pid] = content.strip()
    print("load collection done", "(inverse)." if inverse else ".")
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
    return queries

def load_qrels(path):
    qrels = {}
    with open(path, 'r') as f:
        for line in tqdm(f):
            qid, _, pid, rel = line.strip().split('\t')
            if int(rel) == 1:
                qrels[qid].append(pid)
    print("load qrels done")
    return qrels

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
