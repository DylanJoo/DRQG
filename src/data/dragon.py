import collections
from tqdm import tqdm
import argparse
import json
from datasets import load_dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def passage_centric_dataset(path):
    print(f"Load data from: {path} ...")
    dataset = load_dataset('json', data_files=path)
    print(f"Number of examples: {len(dataset['train'])}")
    return dataset

def convert_to_passage_centric(args):
    """
    Use Dragon synthesized query dataset, which the queries are generated by doc2t5query.
    Each (query-centric) examples has 10 positive and 10 negative passages.
    """
    from data_utils import load_collection
    triplet = collections.defaultdict(dict)
    collection = load_collection(args.collection)

    mms = MinMaxScaler(feature_range=(0, 1))

    with open(args.input_jsonl, 'r') as f:
        for line in tqdm(f):
            example = json.loads(line.strip())
            query = example['question']

            # top 1 - 10
            for docid_score in example['positive_ctxs']:
                docid, score = str(docid_score['docidx']), docid_score['score']
                if docid not in triplet:
                    triplet[docid]['positive'] = list()
                    triplet[docid]['negative'] = list()
                triplet[docid]['positive'].append((query, score))

            # top 46 - 50 
            for docid_score in example['hard_negative_ctxs']:
                docid, score = str(docid_score['docidx']), docid_score['score']
                if docid not in triplet:
                    triplet[docid]['positive'] = list()
                    triplet[docid]['negative'] = list()
                triplet[docid]['negative'].append((query, score))

    # calculate stats
    pos, neg = [], []
    for docid in triplet:
        pos.append(len(triplet[docid]['positive']))
        neg.append(len(triplet[docid]['negative']))

    print(f"# passages: {len(triplet)}")
    print(f"# avg. positive (query): {np.mean(pos)}")
    print(f"# avg. negative (query): {np.mean(neg)}")
    print(f"# passages has zero positive: {(np.array(pos) == 0).sum()}")
    print(f"# passages has zero negative: {(np.array(neg) == 0).sum()}")

    # reverse the mapping for output jsonl
    with open(args.output_jsonl, 'w') as f:
        for docid, queries in triplet.items():
            positives = list(queries['positive'])
            negatives = list(queries['negative'])
            n_pos = len(positives)
            n_neg = len(negatives)
            # sort them
            positives = sorted(positives, key=lambda x: x[1], reverse=True)[:args.max_n]
            negatives = sorted(negatives, key=lambda x: x[1], reverse=True)[:args.max_n]

            # Setting1: neatives inner-join 
            # each p contains at most n_neg instances with same positive
            if 'v1' in args.output_jsonl:
                if (n_pos > 0) and (n_pos < n_neg):
                    positives = (positives * n_neg)[:n_neg]

            # Setting0: inner join (each p contains at most min(n_pos, n_neg))
            if 'v0' in args.output_jsonl or 'v1' in args.output_jsonl:
                for pos, neg in zip(positives, negatives):
                    f.write(json.dumps({
                        "passage": collection[docid],
                        "positive": pos, 
                        "negative": neg
                    }, ensure_ascii=False)+'\n')

            # Setting-L: join a list of postives and negative. 
            # Additionally, put them into a list for making sure there are in-batch
            if 'vL' in args.output_jsonl:
                if n_pos >= args.min_n and n_neg >= args.min_n:
                    p_text, p_score = list(zip(*positives))
                    n_text, n_score = list(zip(*negatives))
                    # normalizing
                    def _reshape(x):
                        return np.array(x).reshape(-1, 1)
                    mms.fit(_reshape(p_score+n_score))
                    p_score_norm = mms.transform(_reshape(p_score)).flatten().tolist()
                    n_score_norm = mms.transform(_reshape(n_score)).flatten().tolist()
                    f.write(json.dumps({
                        "passage": collection[docid],
                        "positive": p_text, 
                        "negative": n_text,
                        "positive_score": p_score_norm, 
                        "negative_score": n_score_norm
                    }, ensure_ascii=False)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, default=None)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, default=None)

    # spec for dragon pseudo datasets
    parser.add_argument("--min_n", type=int, default=1, 
            help='the minumum number of obtained negative query.')
    parser.add_argument("--max_n", type=int, default=10, 
            help='the minumum number of obtained negative query.')
    args = parser.parse_args()

    convert_to_passage_centric(args)