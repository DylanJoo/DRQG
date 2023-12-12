import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from evaluation import READEval
from datasets import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_jsonl")
    parser.add_argument("--prediction")
    parser.add_argument("--output_jsonl", default=None, type=str)
    # evaluator
    parser.add_argument("--encoder_name", default=None)
    parser.add_argument("--regressor_name", default=None)
    parser.add_argument("--reranker_name", default=None)
    parser.add_argument("--device", default='cuda', type=str)
    # generation config
    parser.add_argument("--batch_size", default=2, type=int)

    args = parser.parse_args()

    # Data
    ## load corpus
    corpus = {}
    with open(args.corpus_jsonl, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())

            if isinstance(item['abstract'], list):
                title = item['title']
                abstract  = " ".join(item['abstract'])
            else:
                title = item['title']
                abstract = item['abstract']
            corpus[item['doc_id']] = title + " " + abstract

    ## load prediction
    data = []
    with open(args.prediction, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())

            data.append({
                "query": item['generated_query'], 
                "passage": corpus[item['doc_id']],
                "score": item['relevance_scores']
            })
    dataset = Dataset.from_list(data)

    # READ Evaluator
    evaluator = READEval(
            dataset=dataset,
            encoder_name=args.encoder_name,
            regressor_name=args.regressor_name,
            reranker_name=args.reranker_name,
            device=args.device,
            generator=None
    )

    # Eavluate diversity
    outputs = {}
    if args.encoder_name is not None:
        diversity = evaluator.evaluate_diversity(
                total_query_group=dataset['query'],
                metrics=('euclidean', 'angular'),
                batch_size=args.batch_size,
        )
        outputs.update(diversity)

    # Evaluate consistency
    if args.regressor_name is not None:
        consistency = evaluator.evaluate_consistency(
                total_query_group=dataset['query'],
                total_passages=dataset['passage'],
                total_scores=dataset['score'],
                batch_size=args.batch_size,
        )
        outputs.update(consistency)

    # Evaluate relevancy
    if args.reranker_name is not None:
        # evaluator.set_monot5_as_ranker() # default as monot5-3b msmarco 10k
        relevancy = evaluator.evaluate_relevancy(
                total_query_group=dataset['query'],
                total_passages=dataset['passage'],
                total_scores=dataset['score'],
                batch_size=args.batch_size,
                select_score=1.0,
        )
        outputs.update(relevancy)
        relevancy = evaluator.evaluate_relevancy(
                total_query_group=dataset['query'],
                total_passages=dataset['passage'],
                total_scores=dataset['score'],
                batch_size=args.batch_size,
                select_score=0.0,
        )
        outputs.update(relevancy)

    # mean values
    printer = f"{args.prediction.replace('.jsonl', '').rsplit('/', 1)[-1]}"
    print(printer)
    for metric in outputs:
        print("{:<10}: \n{} | {} | {} | {}".format(
            metric,
            np.mean(outputs[metric]).round(4),
            np.std(outputs[metric]).round(4),
            np.min(outputs[metric]).round(4),
            np.max(outputs[metric]).round(4)
        ))

    if args.output_jsonl is not None:
        with open(args.output_jsonl, 'w') as f:
            f.write(json.dumps(outputs))
