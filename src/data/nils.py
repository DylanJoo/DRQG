import collections
from tqdm import tqdm
import argparse
import json
import pickle
from datasets import load_dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def passage_centric_dataset(path):
    print(f"Load data from: {path} ...")
    dataset = load_dataset('json', data_files=path)
    print(f"Number of examples: {len(dataset['train'])}")
    return dataset

def _reshape(x):
    return np.array(x).reshape(-1, 1)

def convert_to_document_centric(args):
    data=pickle.load(open(args.input_pkl, 'rb'))

    from data_utils import load_collection, load_queries
    triplet = collections.defaultdict(dict)
    collection = load_collection(args.collection)
    queries = load_queries(args.queries)

    mms = MinMaxScaler(feature_range=(0, 1))

    for qid, ranklist in tqdm(data.items()):
        query = queries[str(qid)]

        for docid, score in ranklist.items():
            if docid not in triplet:
                triplet[docid]['positive'] = []
                triplet[docid]['negative'] = []

            if score <= 0:
                triplet[docid]['negative'].append((query, score))
            else:
                triplet[docid]['positive'].append((query, score))

    # calculate stats
    pos, neg = [], []
    for docid in triplet:
        pos.append(len(triplet[docid]['positive']))
        neg.append(len(triplet[docid]['negative']))

    with open(f"{args.output_jsonl.replace('json', 'stats.txt')}", 'w') as fstat:
        fstat.write(f"# passages: {len(triplet)}\n")
        fstat.write(f"# avg. positive (query): {np.mean(pos)}\n")
        fstat.write(f"# avg. negative (query): {np.mean(neg)}\\n")
        fstat.write(f"# passages has zero positive: {(np.array(pos) == 0).sum()}\n")
        fstat.write(f"# passages has zero negative: {(np.array(neg) == 0).sum()}\n")

    # reverse the mapping for output jsonl
    with open(args.output_jsonl, 'w') as f:
        for docid, queries in tqdm(triplet.items()):
            positives = list(queries['positive'])
            negatives = list(queries['negative'])
            n_pos = len(positives)
            n_neg = len(negatives)
            # sort them 
            positives = sorted(positives, key=lambda x: x[1], reverse=True)
            negatives = sorted(negatives, key=lambda x: x[1], reverse=True)

            if n_pos >= args.min_n and n_neg >= args.min_n:
                p_text, p_score = list(zip(*positives))
                n_text, n_score = list(zip(*negatives))
                mms.fit(_reshape(p_score+n_score))
                f.write(json.dumps({
                    "passage": collection[str(docid)],
                    "positive": p_text[:args.max_n], 
                    "negative": n_text[:args.max_n],
                    "positive_score": mms.transform(_reshape(p_score[:args.max_n])).flatten().tolist(),
                    "negative_score": mms.transform(_reshape(n_score[:args.max_n])).flatten().tolist(),
                }, ensure_ascii=False)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, default=None)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--queries", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, default=None)

    # spec for dragon pseudo datasets
    parser.add_argument("--min_n", type=int, default=1, 
            help='the minumum number of obtained negative query.')
    parser.add_argument("--max_n", type=int, default=10, 
            help='the maximun number of obtained negative query.')
    args = parser.parse_args()

    convert_to_document_centric(args)
