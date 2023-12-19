import argparse
import json
from datasets import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # load model
    parser.add_argument("--jsonl", default='sample.jsonl', type=str)
    parser.add_argument("--corpus", default='/home/jhju/datasets/scifact/corpus.jsonl', type=str)
    args = parser.parse_args()

    corpus = load_dataset('json', data_files=args.corpus)['train']

    with open(args.jsonl, 'r') as fin:

        for i, line in enumerate(fin):
            result = json.loads(line.strip())
            scores = result['relevance_scores']
            queries = result['generated_query']

            if 'marco' in args.corpus:
                print(f"{corpus[i]['passage']}\n>>")
                positive = corpus[i]['positive'][:2]
                positive_score = corpus[i]['positive_score'][:2]
                negative = corpus[i]['negative'][:2]
                negative_score = corpus[i]['negative_score'][:2]
                for (q, s) in zip(positive, positive_score):
                    s = round(s, 2)
                    print(f"{s:<3} {q}")
                for (q, s) in zip(negative, negative_score):
                    s = round(s, 2)
                    print(f"{s:<3} {q}")

            if 'sci' in args.corpus:
                print(f"{corpus[i]['title']}\n>>")
                print(" | ".join(corpus[i]['abstract']))

            print(">>")
            for (q, s) in zip(queries, scores):
                s = round(s, 2)
                print(f"{s:<3} {q}")
            print()
