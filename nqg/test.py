import argparse
import msmarco 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplet", default="/home/jhju/datasets/triples.train.small.tsv", type=str)
    parser.add_argument("--collection", default="/tmp2/jhju/datasets/msmarco-psgs/collection.tsv", type=str)
    parser.add_argument("--queries", default="/tmp2/jhju/datasets/msmarco-psgs/queries.train.tsv", type=str)
    parser.add_argument("--joinbynegative", default=False, action='store_true')
    parser.add_argument("--p_aligned_triplet", default='/home/jhju/datasets/triples.train.small.v1.jsonl', type=str)
    args = parser.parse_args()

    args.joinbynegative = True
    dataset = msmarco.passage_aligned_triplet_dataset(args)
