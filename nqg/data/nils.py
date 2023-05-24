import collections
from tqdm import tqdm
import argparse
import json
import pickle
from datasets import load_dataset
import numpy as np

def passage_centric_dataset(path):
    print(f"Load data from: {path} ...")
    dataset = load_dataset('json', data_files=path)
    print(f"Number of examples: {len(dataset['train'])}")
    return dataset

def convert_to_passage_centric(args):
    data=pickle.load(open(args.input_pkl, 'rb'))

    from data_utils import load_collection, load_queries
    triplet = collections.defaultdict(dict)
    collection = load_collection(args.collection)
    queries = load_queries(args.queries)

    for qid, ranklist in tqdm(data.items()):
        query = queries[str(qid)]

        for pid, score in ranklist.items():
            if pid not in triplet:
                triplet[pid]['positive'] = []
                triplet[pid]['negative'] = []

            if score <= 0:
                triplet[pid]['negative'].append((query, score))
            else:
                triplet[pid]['positive'].append((query, score))

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
    with open(args.output_jsonl, 'w') as f:
        for pid, queries in triplet.items():
            positives = list(queries['positive'])
            negatives = list(queries['negative'])
            n_pos = len(positives)
            n_neg = len(negatives)
            # sort them 
            positives = sorted(positives, key=lambda x: x[1], reverse=True)[:args.max_n]
            negatives = sorted(negatives, key=lambda x: x[1], reverse=True)[:args.max_n]

            # Setting-L: join a list of postives and negative. 
            # Additionally, put them into a list for making sure there are in-batch
            if 'vL' in args.output_jsonl:
                if n_pos >= args.min_n and n_neg >= args.min_n:
                    p_text, p_score = list(zip(*positives))
                    n_text, n_score = list(zip(*negatives))
                    f.write(json.dumps({
                        "passage": collection[str(pid)],
                        "positive": p_text, 
                        "positive_score": p_score, 
                        "negative": n_text,
                        "negative_score": n_score
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

    convert_to_passage_centric(args)