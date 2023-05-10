import json
import copy
import torch
import argparse
import collections
from tqdm import tqdm
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from transformers import T5ForConditionalGeneration
from datacollator import DataCollatorForT5VQG
from utils import interpolate

from models import T5VQGV0, T5VQGV1, T5PQG, T5VQGSPT, T5VQGDEV
from models import VAE_CONFIG
class QuestionGenerator:

    def __init__(self, 
                 model, 
                 tokenizer, 
                 generation_type, 
                 generation_config):

        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.generation_type = generation_type
        self.flags = generation_config.pop('flags', [])
        self.generation_config = generation_config

        self.total_n_samples = model.n_samples if model.n_samples else 5
        #`num_beams`, `do_sample`, `top_k`

    def __call__(self, **batch):
        """
        This generation is for variational inference.

        Returns
        -------
        Each return object is a dictionary of prediction;
        The key(index) represents the batch examples.
        """
        batch_size = batch['input_ids'].size(0)

        ## TODO sepeare each type of generation: 
        ### (1) latent z reconstriction
        ### (2) soft prompting
        if isinstance(self.model, T5VQGV1):
            enc_output = self.model.encoder(**batch)
            hidden_states = enc_output[0]
            embeds = copy.deepcopy(hidden_states[:, :1, :])

            if self.generation_type == 'gaussian':
                hidden_states_prime = self._gaussian_encoding(embeds, hidden_states)
            elif self.generation_type == 'interpolation':
                hidden_states_prime = self._interpolate_encoding(embeds)

            enc_output.last_hidden_state = hidden_states_prime
            outputs = self.model.generate(
                    encoder_outputs=enc_output,
                    **self.generation_config
            )
        elif isinstance(self.model, T5VQGSPT):
            outputs = self.model.generate(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    **self.generation_config
            )

        ## Convert token to texts
        texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        print("\n".join(texts))

        ## Collect outputs
        output_dict = collections.defaultdict(list)
        for i in range(len(texts)):
            output_dict[i % batch_size].append(texts[i])

        return output_dict

    def _gaussian_encoding(self, embeds, hidden_states):
        """ In the following three condition, 
        the first two are for VQG_v0; the last is for VQG_v1
        In VQG_v0 and v1, n_sample is the amounts of left or right.
        """
        n_side = (self.total_n_samples - 1) // 2
        std_list = list(range(-(self.n_side), self.n_side+1, 1))
        self.total_n_samples = len(self.flags) * len(std_list)

        zs = []
        ## generate the latent z
        for flag in self.flags:
            if flag == 'positive':
                mean = self.model.hidden2pmean(embeds)
                std = self.model.hidden2plogv(embeds)
                zs += [mean+(std*n) for n in std_list]
            elif flag == 'negative':
                mean = self.model.hidden2nmean(embeds)
                std = self.model.hidden2nlogv(embeds)
                zs += [mean+(std*n) for n in std_list]
            elif flag == 'polarity':
                mean = self.model.hidden2mean(embeds)
                std = self.model.hidden2logv(embeds)
                zs += [mean+(std*n) for n in std_list]
            else:
                raise ValueError(f"The value flag, {flag} is invalid.")

        ## transform into hidden
        resid_z = self.model.latent2hidden(torch.cat(zs, 0))

        ## residual adding
        zeros = torch.zeros(resid_z.size(0), 
                            resid_z.size(1)-1, 
                            resid_z.size(2)).to(embeds.device)
        resid = torch.cat((resid_z, zeros), 1)
        h_prime = hidden_states.repeat((self.total_n_samples, 1, 1)) + resid
        return h_prime

    def _interploate_encoding(self):
        pass

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

    # variational inference (only latent_size is required)
    parser.add_argument("--latent_size", default=128, type=int)

    ## the unused parameters
    parser.add_argument("--k", default=0.0025, type=float) 
    parser.add_argument("--x0", default=2500, type=int)
    parser.add_argument("--annealing_fn", default='logistic')

    # generation type
    parser.add_argument("--generation_type", default='gaussian', type=str)

    # generation config
    parser.add_argument("--n_soft_prompts", default=20, type=int)
    parser.add_argument("--n_samples", default=0, type=int)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--max_q_length", default=20, type=int)
    parser.add_argument("--max_p_length", default=256, type=int)
    parser.add_argument("--do_sample", default=False, action='store_true')
    parser.add_argument("--top_k", default=10, type=int)

    args = parser.parse_args()
    print(args)

    vae_config = VAE_CONFIG(
            latent_size=args.latent_size, 
            n_soft_prompts=args.n_soft_prompts
    )
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    MODELS = {
            'vqgv0': T5VQGV0, 
            'vqgv1': T5VQGV1, 
            'pqg': T5PQG, 
            'vqgspt': T5VQGSPT,
            'vqgdev': T5VQGDEV
    }

    for key in MODELS:
        if key in args.model_path.lower():
            model = MODELS[key].from_pretrained(
                    pretrained_model_name_or_path=args.model_path,
                    config=config,
                    vae_config=vae_config,
                    tokenizer=tokenizer, 
            ).to(args.device).eval()

    # load dataset
    dataset = load_dataset("json", data_files=args.input_jsonl)['train']

    from datacollator import DataCollatorForT5Dev
    data_collator = DataCollatorForT5Dev(
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

    # new a generator 
    ## generation config
    generator_config = {
            'num_beams': args.num_beams,
            'max_length': args.max_q_length,
            'do_sample': args.do_sample,
            'top_k': args.top_k,
    }
    generator_config['flags'] = args.flags

    generator = QuestionGenerator(
            model=model, 
            tokenizer=tokenizer, 
            generation_type=args.generation_type, 
            generation_config=generator_config
    )

    # prediction
    for batch in tqdm(dataloader):
        output_dict = {i: 
                {"passage": p, 
                 "positive_truth": pq, 
                 "negative_truth": nq} \
                for i, (p, pq, nq) in enumerate(
                    zip(batch.pop('passage'), batch.pop('positive'), batch.pop('negative'))
                )
        }

        for k in batch:
            batch[k] = batch[k].to(args.device)

        # forward and generate
        with torch.no_grad():
            predictions = generator(**batch)
            N = generator.total_n_samples
            for i, output in output_dict.items():
                # append queries with diff modes of this passage
                for flag in args.flags:
                    output.update({f"{flag}": predictions[i][:N]})
                    del predictions[i][:N]

                f.write(json.dumps(output)+'\n')
    f.close()


    # transform to good read version
    from utils import transform_pred_to_good_read
    transform_pred_to_good_read(
            args.output_jsonl, args.output_jsonl.replace('jsonl', 'txt')
    )

