import torch
import argparse
from tqdm import tqdm
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from datacollator import DataCollatorForT5VQG
from models import T5PQG

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # load model
    parser.add_argument("--model_name")
    parser.add_argument("--model_path")
    parser.add_argument("--beam_size")
    parser.add_argument("--input_jsonl")
    parser.add_argument("--output_jsonl")
    parser.add_argument("--positive", action='store_true', default=False)
    parser.add_argument("--negative", action='store_true', default=False)
    parser.add_argument("--device", defualt='cuda')
    parser.add_argument("--batch_size", defualt=2)

    # generation config
    parser.add_argument("--beam_size", defualt=5)
    parser.add_argument("--max_length", defualt=20)

    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5PQG.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            config=config,
    ).to(args.device).eval()

    # load dataset
    dataset = load_dataset("json", data_files=args.input_jsonl)

    from datacollator import DataCollatorForT5PQG
    data_collator = DataCollatorForT5PQG(
            tokenizer=tokenizer, 
            padding=True,
            return_tensors='pt',
            is_train=False,
            is_eval=True # to check the ground truth
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=data_collator
    )

    # new a writer
    f = open(args.output_jsonl, 'w')

    # prediction
    for batch, batch1, batch0 in tqdm(dataloader):
        output_dict = {i: {"passage": p, "positive_truth": pq, "negative_truth": nq} \
                for i, (p, pq, nq) in enumerate(zip(batch.pop('passage'), batch.pop('positive'), batch.pop('negative')))
        }
        for k in batch1:
            batch1[k] = batch1[k].to(device)
        for k in batch0:
            batch0[k] = batch0[k].to(device)

        # forward and generate
        with torch.no_grad():
            output1 = model.generate(
                    **batch1,
                    num_beams=args.beam_size,
                    max_length=args.max_length
            )
            output0 = model.generate(
                    **batch0,
                    num_beams=args.beam_size,
                    max_length=args.max_length
            )

            for i in range(len(output_dict)):
                output_dict[i]['positive'] = tokenizer.decode(output1[i], skip_special_tokens=True)
                output_dict[i]['negative'] = tokenizer.decode(output0[i], skip_special_tokens=True) 

            for k, v in output_dict.items():
                f.write(json.dumps(v)+'\n')

    f.close()
