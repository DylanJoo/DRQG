from datasets import load_dataset
import collections
from tqdm import tqdm
import argparse
import json

def passage_centric_dataset(path):
    print(f"Load data from: {path}...")
    dataset = load_dataset('json', data_files=path)
    print(f"Number of examples: {len(dataset['train'])}")
    return dataset

def convert_to_passage_centric(args):
    data = json.load(open(args.input_json, 'r'))['data']
    collection = {}

    f = open(args.output_jsonl, 'w') 

    for example in tqdm(data):
        title = example['title']
        paragraphs = example['paragraphs']
        for i, paragraph in enumerate(paragraphs):
            collection[f"{title}_{i}"] = paragraph['context'].strip()
            questions = []
            for qa in paragraph['qas']:
                questions.append(qa['question'])
                # answer = qa['answer']['text']

            f.write(json.dumps({
                "passage": collection[f"{title}_{i}"],
                "positive": questions, 
                "negative": ""
            }, ensure_ascii=False)+'\n')

    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, default=None)
    args = parser.parse_args()

    convert_to_passage_centric(args)
