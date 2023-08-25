import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # load model
    parser.add_argument("--jsonl", default='sample.jsonl', type=str)
    args = parser.parse_args()

    with open(args.jsonl, 'r') as fin:

        for line in fin:
            result = json.loads(line.strip())
            scores = result['condition_score']
            queries = result['generated_query']
            print(f"{result['passage']}\n>>")
            for (q, s) in zip(queries, scores):
                s = round(s, 2)
                print(f"{s:<3} {q}")
            print()
