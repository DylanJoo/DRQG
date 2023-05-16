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

    def __call__(self, **batch):
        batch_size = batch['input_ids'].size(0)

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
        else:
            outputs = self.model.generate(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    return_dict_in_generate=True,
                    output_scores=True,
                    **self.generation_config
            )

        ## Convert token to texts
        texts = []
        for output in outputs.sequences:
            text=self.tokenizer.decode(output, skip_special_tokens=True)
            texts.append(text)

        output_dict = collections.defaultdict(list)
        for i in range(len(texts)):
            output_dict[i % batch_size].append(texts[i])

        return output_dict

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

    ## the model parameters
    parser.add_argument("--k", default=0.0025, type=float) 
    parser.add_argument("--x0", default=2500, type=int)
    parser.add_argument("--annealing_fn", default='logistic')

    # generation type
    parser.add_argument("--generation_type", default='gaussian', type=str)

    # generation config
    parser.add_argument("--n_soft_prompts", default=20, type=int)
    parser.add_argument("--n_side_tail", default=0, type=int)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--max_q_length", default=100, type=int)
    parser.add_argument("--max_p_length", default=256, type=int)
    parser.add_argument("--do_sample", default=False, action='store_true')
    parser.add_argument("--top_k", default=10, type=int)

    args = parser.parse_args()

    # Config and tokenizer
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Model
    from models import BartQG, T5QG, BartVQG, VQG_CONFIG, T5VQGV0, T5VQGV1
    MODELS = {'bartqg': BartQG, 't5qg': T5QG, 'bartvqg': BartVQG}
    vqg_config = VQG_CONFIG(
            latent_size=args.latent_size, 
            n_soft_prompts=args.n_soft_prompts,
            n_side=args.n_side_tail,
            pooling='adaptive' if 'ada' in args.model_path else 'static'
    )

    for key in MODELS:
        if key in args.model_path.lower():
            model_key = key

    model = MODELS[model_key].from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            config=config,
            vqg_config=vqg_config
    ).to(args.device).eval()
    model.set_tokenizer(tokenizer)

    # Data: dataset
    dataset = load_dataset("json", data_files=args.input_jsonl)['train']

    # Data: datacollator
    from datacollator import DataCollatorForVQGSPT
    data_collator = DataCollatorForVQGSPT(
            tokenizer=tokenizer, 
            padding=True,
            return_tensors='pt',
            max_p_length=args.max_p_length,
            max_q_length=args.max_q_length,
            is_eval=True # to check the ground truth
    )

    # Data: dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=data_collator
    )

    # Generation: writer
    f = open(args.output_jsonl, 'w')

    # Generation: config
    generator_config = {
            'num_beams': args.num_beams,
            'max_length': args.max_q_length,
            'do_sample': args.do_sample,
            'top_k': args.top_k,
    }
    generator_config['flags'] = args.flags

    # Generation: generator
    generator = QuestionGenerator(
            model=model, 
            tokenizer=tokenizer, 
            generation_type=args.generation_type, 
            generation_config=generator_config
    )

    for batch in tqdm(dataloader):
        output_dict = {i: {
            "passage": p, "positive_truth": pq, "negative_truth": nq
            } for i, (p, pq, nq) in enumerate(
                zip(batch.pop('passage'), 
                    batch.pop('positive'), 
                    batch.pop('negative'))
        )}

        for k in batch:
            batch[k] = batch[k].to(args.device)

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

