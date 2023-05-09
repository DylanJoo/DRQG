import json
import copy
import torch
import argparse
from tqdm import tqdm
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from datacollator import DataCollatorForT5VQG
from models2 import T5VQG
from utils import interpolate

def interpolated_generation(
        positive,
        negative,
        model, 
        hidden_states, 
        tokenizer,
        interpolate_n=None, 
    ):
    e_embed = hidden_states[:, :1, :]
    N = interpolate_n

    # reparameterize with none token
    none_ = tokenizer('<extra_id_10>', return_tensors='pt').to(e_embed.device)
    none_ = model.encoder(**none_)[0][:, :1, :]

    # reparameterize
    if positive:
        A = model.hidden2pmean(none_)
        B = model.hidden2pmean(e_embed)
    if negative:
        A = model.hidden2nmean(none_)
        B = model.hidden2nmean(e_embed)

    # decoding 1: interpolation with one endpoints
    zs = interpolate(A, B, N)

    ### So far, arranging the first dimension (batch) as batch x len(std_list)
    z = torch.cat(zs, 0)
    e_embed_new = model.latent2hidden(z) 
    zeros = torch.zeros(
            hidden_states.size(0) * N,
            hidden_states.size(1)-1, 
            hidden_states.size(2)
    ).to(e_embed.device)
    resid = torch.cat((e_embed_new, zeros), 1)
    return resid + hidden_states.repeat((N, 1, 1))


def parameterized_generation(
        positive, 
        model, 
        hidden_states, 
        std_list=None, 
        debug=0
    ):

    if debug == 1:
        e_embed = hidden_states[:, :, :]
    else:
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
    if debug == 0:
        z = torch.cat(zs, 0)
        e_embed_new = model.latent2hidden(z) 
        zeros = torch.zeros(
                hidden_states.size(0)*len(std_list), 
                hidden_states.size(1)-1, 
                hidden_states.size(2)
        ).to(e_embed.device)
        resid = torch.cat((e_embed_new, zeros), 1)
        return resid + hidden_states.repeat((N, 1, 1))

    elif debug == 1:
        z = torch.cat(zs, 0)
        hidden_new = model.latent2hidden(z) 
        return hidden_new
    
    elif debug == 2:
        z = torch.cat(zs, 0)
        e_embed_new = model.latent2hidden(z) 
        return torch.cat((e_embed_new, hidden_states.repeat((N, 1, 1))), 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # load model
    parser.add_argument("--model_name")
    parser.add_argument("--model_path")
    parser.add_argument("--input_jsonl")
    parser.add_argument("--output_jsonl")
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--flags", nargs='+', type=str)
    parser.add_argument("--debug", default=0, type=int)

    # variational inference (only latent_size is required)
    parser.add_argument("--latent_size", default=256, type=int)
    parser.add_argument("--sampling", default='gaussian', type=str)

    ## the unused parameters
    parser.add_argument("--k", default=0.0025, type=float) 
    parser.add_argument("--x0", default=2500, type=int)
    parser.add_argument("--annealing_fn", default='logistic')

    # generation config
    parser.add_argument("--n_samples", default=0, type=int)
    parser.add_argument("--beam_size", default=5, type=int)
    parser.add_argument("--max_q_length", default=20, type=int)
    parser.add_argument("--max_p_length", default=256, type=int)
    parser.add_argument("--do_sample", default=False, action='store_true')
    parser.add_argument("--top_k", default=10, type=int)

    args = parser.parse_args()

    # load configuration
    @dataclass
    class VaeConfig:
        latent_size: int
        annealing_fn: str = field(default='logistic') 
        k: float = field(default=0.0025)
        x0: int = field(default=2500)

    vae_config = VaeConfig(args.latent_size)

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5VQG.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            config=config,
            vae_config=vae_config,
            tokenizer=tokenizer,
            debug=args.debug
    ).to(args.device).eval()

    # load dataset
    dataset = load_dataset("json", data_files=args.input_jsonl)['train']

    from datacollator import DataCollatorForT5VQG
    data_collator = DataCollatorForT5VQG(
            tokenizer=tokenizer, 
            padding=True,
            return_tensors='pt',
            max_length=args.max_p_length,
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
    for batch in tqdm(dataloader):
        output_dict = {i: {"passage": p, "positive_truth": pq, "negative_truth": nq} \
                for i, (p, pq, nq) in enumerate(zip(batch.pop('passage'), batch.pop('positive'), batch.pop('negative')))
        }

        for k in batch:
            batch[k] = batch[k].to(args.device)

        # forward and generate
        with torch.no_grad():
            ## encode 
            enc_output = model.encoder(**batch)
            bs = enc_output[0].size(0)
            hidden_states = copy.deepcopy(enc_output[0])

            ## Setting1: parameterized generation
            std_list = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
            # std_list = [-2, -1, 0, 1, 2]
            N = len(std_list) if args.n_samples == 0 else args.n_samples

            if 'positive' in args.flags:
                if args.sampling == 'gaussian':
                    enc_output.last_hidden_state = parameterized_generation(
                            True, model, hidden_states, std_list, args.debug
                    )
                if args.sampling == 'interpolate_none':
                    enc_output.last_hidden_state = interpolated_generation(
                            True, None, model, hidden_states, tokenizer, args.n_samples, args.debug
                    )
                outputs = model.generate(
                        encoder_outputs=enc_output, 
                        num_beams=args.beam_size,
                        max_length=args.max_q_length,
                        do_sample=args.do_sample,
                        top_k=args.top_k
                )
                for i in range(bs):
                    output_dict[i]['positive'] = [\
                            tokenizer.decode(
                                outputs[i+(j*bs)], skip_special_tokens=True
                            ) for j in range(N)
                    ]

                # making sure that the hidden states are copied not in reference.
            if 'negative' in args.flags:
                if args.sampling == 'gaussian':
                    enc_output.last_hidden_state = parameterized_generation(
                            False, model, hidden_states, std_list, args.debug
                    )
                if args.sampling == 'interpolate_none':
                    enc_output.last_hidden_state = interpolated_generation(
                            None, True, model, hidden_states, tokenizer, args.n_samples, args.debug
                    )

                outputs = model.generate(
                        encoder_outputs=enc_output, 
                        num_beams=args.beam_size,
                        max_length=args.max_q_length,
                        do_sample=args.do_sample,
                        top_k=args.top_k
                )
                for i in range(len(output_dict)):
                    output_dict[i]['negative'] = [\
                            tokenizer.decode(
                                outputs[i+(j*bs)], skip_special_tokens=True
                            ) for j in range(N)
                    ]

            for k, v in output_dict.items():
                f.write(json.dumps(v)+'\n')
    f.close()


    # transform to good read version
    from utils import transform_pred_to_good_read
    transform_pred_to_good_read(
            args.output_jsonl, args.output_jsonl.replace('jsonl', 'txt')
    )
