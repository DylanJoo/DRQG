# {"question":"which pope has the middle name gregory","subsets":["L4"],"answer":["Pope Gregory I"],"answers":[{"passage_id":"2581755","offset":565,"text":"Pope Gregory I","extractor":"L"}],"passage_score":"-0.06620407104492188"}
import collections
from tqdm import tqdm
import argparse
import json
from datasets import load_dataset
import numpy as np

def convert_to_passage_centric(args):
    from data_utils import load_collection
    triplet = collections.defaultdict(dict)
    collection = load_collection(args.collection)

    with open(args.input_jsonl, 'r') as f:
        for line in tqdm(f):
            example = json.loads(line.strip())
            query = example['question']
            answer = example['answer']
            pids = [a['passage_id'] for a in example['answers']]

            for pid in pids:
                if pid not in triplet:
                    triplet[pid]['positive'] = set()
                    triplet[pid]['negative'] = set()
                triplet[pid]['positive'].update([query])

            # top 46 - 50 
            for docid_score in example['hard_negative_ctxs']:
                docid, score = str(docid_score['docidx']), docid_score['score']
                if docid not in triplet:
                    triplet[docid]['positive'] = set()
                    triplet[docid]['negative'] = set()
                triplet[docid]['negative'].update([query])

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
                    f.write(json.dumps({
                        "passage": collection[docid],
                        "positive": positives, 
                        "negative": negatives
                    }, ensure_ascii=False)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, default=None)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, default=None)

    # spec for dragon pseudo datasets
    parser.add_argument("--min_n", type=int, default=1, 
            help='the minumum number of obtained negative query.')
    args = parser.parse_args()

    convert_to_passage_centric(args)