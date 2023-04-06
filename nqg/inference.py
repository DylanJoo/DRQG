import torch
import argparse
from tqdm import tqdm
from dataclasses import dataclass, field
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer
from datacollator import DataCollatorForT5VQG
from models import T5VQG

def parameterized_generation(positive, model, e_embed, std_list=None):
    # reparameterize
    if positive:
        mean = model.hidden2pmean(e_embed)
        logv = model.hidden2plogv(e_embed)
    else:
        mean = model.hidden2nmean(e_embed)
        logv = model.hidden2nlogv(e_embed)

    std = torch.exp(0.5 * logv)
    N = len(std_list)
    zs = [mean+(std*n) for n in std_list]

    ### So far, arranging the first dimension (batch) as batch x len(std_list)
    z = torch.cat(zs, 0)
    e_embed = model.latent2hidden(z) 
    zeros = torch.zeros(
            e_embed_size(0)*len(std_list), e_embed.size(1)-1, e_embed.size(2)
    ).to(e_embed.device)
    resid = torch.cat((e_embed, zeros), 1)
    return resid + hidden_states.repeat((N, 1, 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # load model
    parser.add_argument("--model_name")
    parser.add_argument("--model_path")
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

    from datacollator import DataCollatorForT5VQG
    data_collator = DataCollatorForT5VQG(
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
            ## encode 
            enc_output = model.encoder(**batch)

            ## Setting1: parameterized generation
            std_list = [-2, -1, 0, 1, 2]
            N = len(std_list)
            if args.positive:
                enc_output.last_hidden_state = parameterized_generation(
                        True, model, e_embed, args.std_list
                )
                outputs = model.generate(
                        encoder_outputs=enc_output, 
                        num_beams=args.beam_size,
                        max_length=args.max_length
                )
                for i in range(len(output_dict)):
                    output_dict[i]['positive'] = [\
                            tokenizer.decode(outputs[i+j]) for j in range(N)
                    ]

            if args.negative:
                enc_output.last_hidden_state = parameterized_generation(
                        False, model, e_embed, args.std_list
                )
                outputs = model.generate(
                        encoder_outputs=enc_output, 
                        num_beams=args.beam_size,
                        max_length=args.max_length
                )
                for i in range(len(output_dict)):
                    output_dict['negative'] = [\
                            tokenizer.decode(outputs[i+j]) for j in range(N)
                    ]

        with open(args.output_jsonl, 'w') as f:
            for k, v in output_dict.items():
                f.write(json.dumps({v})+'\n')
