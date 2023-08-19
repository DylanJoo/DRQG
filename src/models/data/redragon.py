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
                    triplet[docid]['qid_score'] = list()
                triplet[docid]['qid_score'].append((query, score))

            # top 46 - 50 
            for docid_score in example['hard_negative_ctxs']:
                docid, score = str(docid_score['docidx']), docid_score['score']
                if docid not in triplet:
                    triplet[docid]['qid_score'] = list()
                triplet[docid]['qid_score'].append((query, score))

    # reverse the mapping for output jsonl
    with open(args.output_jsonl, 'w') as f:
        for docid, queries in tqdm(triplet.items()):
            qids_scores = list(queries['qid_score'])
            n = len(qids_scores)
            examples = sorted(qids_scores, key=lambda x: x[1], reverse=True)

            if n >= args.min_n:
                def _reshape(x):
                    return np.array(x).reshape(-1, 1)
                q_examples, score_examples = list(zip(*examples))
                score_norm = mms.fit_transform(_reshape(score_examples)).flatten().tolist()

                p_text, n_text = q_examples[:n//2][:args.max_n], q_examples[n//2:]
                p_score, n_score = score_norm[:n//2][:args.max_n], score_norm[n//2:]

                f.write(json.dumps({
                    "passage": collection[docid],
                    "positive": p_text, 
                    "negative": n_text,
                    "positive_score": p_score, 
                    "negative_score": n_score
                }, ensure_ascii=False)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, default=None)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, default=None)
    # spec for dragon pseudo datasets
    parser.add_argument("--min_n", type=int, default=2, 
            help='the minumum number of obtained negative query.')
    parser.add_argument("--max_n", type=int, default=5, 
            help='the minumum number of obtained negative query.')
    args = parser.parse_args()

    convert_to_passage_centric(args)
