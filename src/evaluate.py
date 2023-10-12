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
    # evaluator
    parser.add_argument("--encoder_name")
    parser.add_argument("--ranker_name")
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
                corpus[item['doc_id']]  = " ".join(item['abstract'])
            else:
                corpus[item['doc_id']] = item['abstract']

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
    print(dataset)

    # READ Evaluator
    evaluator = READEval(
            dataset=dataset,
            encoder_name=args.encoder_name,
            ranker_name=args.ranker_name,
            device=args.device,
            generator=None
    )

    # Eavluate diversity
    outputs = {}
    diversity = evaluator.evaluate_diversity(
            total_query_group=dataset['query'],
            metrics=('euclidean', 'angular'),
            batch_size=args.batch_size,
    )
    outputs.update(diversity)

    # mean values
    print("==== Evaluation Results ====\n")
    for m in outputs:
        print(f"# {m:<10} mean: ",np.mean(outputs[m]).round(4))
        print(f"# {m:<10} std : ", np.std(outputs[m]).round(4))
        print(f"# {m:<10} max : ", np.max(outputs[m]).round(4))
        print(f"# {m:<10} min : ", np.min(outputs[m]).round(4))
    print("\n============================")
