import json
import copy
import torch
import argparse
from tqdm import tqdm
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from datacollator import DataCollatorForT5VQG
from models import T5VQG
from utils import interpolate

def interpolated_generation(
        positive, 
        model, 
        hidden_states, 
        interpolate_n=none, 
    ):
    e_embed = hidden_states[:, :1, :]
    # reparameterize
    if positive:
        mean = model.hidden2pmean(e_embed)
    else:
        mean = model.hidden2nmean(e_embed)

    ### So far, arranging the first dimension (batch) as batch x len(std_list)
    z = torch.cat(zs, 0)
    e_embed_new = model.latent2hidden(z) 
    zeros = torch.zeros(
            hidden_states.size(0)*len(std_list), 
            hidden_states.size(1)-1, 
            hidden_states.size(2)
    ).to(e_embed.device)
    resid = torch.cat((e_embed_new, zeros), 1)
    return resid + hidden_states.repeat((N, 1, 1))


def parameterized_generation(
        positive, 
        model, 
        hidden_states, 
        std_list=none, 
    ):

    e_embed = hidden_states[:, :1, :]
    # reparameterize
    if positive:
        mean = model.hidden2pmean(e_embed)
        logv = model.hidden2plogv(e_embed)
    else:
        mean = model.hidden2nmean(e_embed)
        logv = model.hidden2nlogv(e_embed)

    # decoding 1: gaussian
    std = torch.exp(0.5 * logv)
    N = len(std_list)
    zs = [mean+(std*n) for n in std_list]

    ### So far, arranging the first dimension (batch) as batch x len(std_list)
    z = torch.cat(zs, 0)
    e_embed_new = model.latent2hidden(z) 
    zeros = torch.zeros(
            hidden_states.size(0)*len(std_list), 
            hidden_states.size(1)-1, 
            hidden_states.size(2)
    ).to(e_embed.device)
    resid = torch.cat((e_embed_new, zeros), 1)
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
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--batch_size", default=2, type=int)

    # variational inference (only latent_size is required)
    parser.add_argument("--latent_size", default=256, type=int)
    parser.add_argument("--decoding_type", default='gaussian', type=str)
    ## the unused parameters
    parser.add_argument("--k", default=0.0025, type=float) 
    parser.add_argument("--x0", default=2500, type=int)
    parser.add_argument("--annealing_fn", default='logistic')

    # generation config
    parser.add_argument("--beam_size", default=5, type=int)
    parser.add_argument("--max_length", default=20, type=int)
    parser.add_argument("--do_sample", default=False, action='store_true')
    parser.add_argument("--top_k", default=10, type=int)

    args = parser.parse_args()

    # load configuration
    vae_config = {"latent_size": args.latent_size,
                  "annealing_fn": args.annealing_fn,
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
    dataset = load_dataset("json", data_files=args.input_jsonl)['train']

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

    # new a writer
    f = open(args.output_jsonl, 'w')

    # prediction
    for batch in tqdm(dataloader):
        output_dict = {i: {"passage": p} for i, p in enumerate(batch.pop('passage'))}
        output = {flag: None for flag in args.flags}

        for k in batch:
            batch[k] = batch[k].to(args.device)

        # forward and generate
        with torch.no_grad():
            ## encode 
            enc_output = model.encoder(**batch)
            hidden_states = copy.deepcopy(enc_output[0])
            bs = enc_output[0].size(0)

            ## Setting1: parameterized generation
            std_list = [-2, -1, 0, 1, 2]
            # std_list = [0, 0, 0, 0, 0]

            if 'positive' in args.flags:
                if decoding_type == 'gaussian':
                    enc_output.last_hidden_state = parameterized_generation(
                            True, model, hidden_states, std_list
                    )
                if decoding_type == 'interpolate':
                    pass
                outputs['positive'] = model.generate(
                        encoder_outputs=enc_output, 
                        num_beams=args.beam_size,
                        max_length=args.max_length,
                        do_sample=args.do_sample,
                        top_k=args.top_k
                )

                # making sure that the hidden states are copied not in reference.
            if 'negative' in args.flags:
                if decoding_type == 'gaussian':
                    enc_output.last_hidden_state = parameterized_generation(
                            False, model, hidden_states, std_list
                    )
                if decoding_type == 'interplolate':
                    pass
                outputs['positive'] = model.generate(
                        encoder_outputs=enc_output, 
                        num_beams=args.beam_size,
                        max_length=args.max_length,
                        do_sample=args.do_sample,
                        top_k=args.top_k
                )

            for k, v in output_dict.items():
                f.write(json.dumps(v)+'\n')
    f.close()
