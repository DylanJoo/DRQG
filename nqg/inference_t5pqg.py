import torch
import argparse
from tqdm import tqdm
from dataclasses import dataclass, field
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer
from datacollator import DataCollatorForT5VQG
from models import T5VQG

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

    # variational inference (only latent_size is required)
    parser.add_argument("--latent_size", defualt=256)
    parser.add_argument("--k", defualt=0.0025)
    parser.add_argument("--x0", defualt=2500)
    parser.add_argument("--annealing_fn", defualt='logistic')

    # generation config
    parser.add_argument("--beam_size", defualt=5)
    parser.add_argument("--max_length", defualt=20)

    args = parser.parse_args()

    # load configuration
    vae_config = {"latent_size": args.latent_size
                  "annealing_fn": args.annealing_fn
                  "k": args.k, 
                  "x0": args.x0}

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5VQG.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            config=config,
            vae_config=vae_config,
            tokenizer=tokenizer
    ).to(args.device).eval()

    # load dataset
    dataset = load_dataset("json", data_files=args.input_jsonl)

    from datacollator import DataCollatorForT5PQG
    data_collator = DataCollatorForT5PQG(
            tokenizer=tokenizer, 
            padding=True,
            return_tensors='pt',
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=data_collator
    )

    # prediction
    for batch in tqdm(dataloader):
        output_dict = {i: {"passage": p} for i, p in enumerate(batch.pop('passage'))}
        for k in batch:
            batch[k] = batch[k].to(device)

        # forward and generate
        with torch.no_grad():
            ## [NOTE] Here arrange the input as batch_size x (positive, negative) in a same batch
            enc_output = model.generate(
                    **batch,
                    num_beams=args.beam_size,
                    max_length=args.max_length
            )

            for i in range(len(output_dict)):
                ## one half as positive
                output_dict[i]['positive'] = tokenizer.decode(outputs[i])
                ## the other as negative
                output_dict[i]['negative'] = tokenizer.decode(outputs[i+N])

        with open(args.output_jsonl, 'w') as f:
            for k, v in output_dict.items():
                f.write(json.dumps({v})+'\n')
