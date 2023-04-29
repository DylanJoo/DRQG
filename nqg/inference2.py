0mport json
0mport copy
import torch
import argparse
import collections
from tqdm import tqdm
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from datacollator import DataCollatorForT5VQG
from models2 import T5VQG
from utils import interpolate

class QuestionGenerator:

    def __init__(self, 
                 model, 
                 tokenizer, 
                 n_samples, 
                 generation_type, 
                 generation_config):

        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.generation_type = generation_type
        self.flags = generation_config.pop('flags', [])
        self.generation_config = generation_config
        #`num_beams`, `do_sample`, `top_k`

    def __call__(self, **batch):
        """
        This generation is for variational inference.

        Returns
        -------
        Each return object is a dictionary of prediction;
        The key(index) represents the batch examples.
        """
        enc_output = self.model.encoder(**batch)
        hidden_states = enc_output[0]
        bs = hidden_states.size(0)
        self.embeds = copy.deepcopy(hidden_states[:, :1, :])

        # Encode the input sequence 
        ## sample with gaussian or interpolation
        if self.generation_type == 'gaussian':
            zs = self._gaussian_encoding()
        elif self.generation_type == 'interpolation':
            zs = self._interpolate_encoding(type=0)

        ## recontruct decoder input
        hidden_states_prime = self._reconstruct_z(
                hidden_states, zs, 'residual_z'
        )
        enc_output.last_hidden_state = hidden_states_prime

        # Decode from the input seqneces (and its variants)
        outputs = self.model.generate(
                encoder_outputs=enc_output,
                **self.generation_config
        )

        ## Convert token to texts
        texts = [self.tokenizer.decode(
            output, skip_special_tokens=True) for output in outputs]

        ## Collect outputs
        output_dict = collections.defaultdict(list)
        for i in range(len(texts)):
            output_dict[i % bs].append(texts[i])

        return output_dict

    def _reconstruct_z(self, h, zs, concat_type):
        """
        h: hidden states; zs: reparemterized latent vector.
        concat_type: 
          - residual_z
            Reparameterized the dummy token and add to original token embeddings. 
            This reconstruction is performed at single layer, 
            which means enc's last or dec's first

        [TODO] All layers
        [TODO] Replace the dummy token (but it lost gradient)
        """
        hz_prime = self.model.latent2hidden(zs)
        if concat_type == 'residual_z':
            zeros = torch.zeros(
                    hz_prime.size(0), h.size(1)-1, hz_prime.size(2)
            ).to(h.device)
            resid = torch.cat((hz_prime, zeros), 1)

            return h.repeat((self.total_n_samples, 1, 1)) + resid

    def _gaussian_encoding(self):
        """ In the following three condition, 
        the first two are for VQG_v0; the last is for VQG_v1
        In VQG_v0 and v1, n_sample is the amounts of left or right.
        """
        std_list = list(range(-(self.n_samples), self.n_samples+1, 1))
        self.total_n_samples = len(self.flags) * len(std_list)
        dec_input_list = []

        for flag in self.flags:
            if flag == 'positive':
                mean = self.model.hidden2pmean(self.embeds)
                std = self.model.hidden2plogv(self.embeds)
                dec_input_list += [mean+(std*n) for n in std_list]
            elif flag == 'negative':
                mean = self.model.hidden2nmean(self.embeds)
                std = self.model.hidden2nlogv(self.embeds)
                dec_input_list += [mean+(std*n) for n in std_list]
            elif flag == 'polarity':
                mean = self.model.hidden2mean(self.embeds)
                std = self.model.hidden2logv(self.embeds)
                dec_input_list += [mean+(std*n) for n in std_list]
            else:
                raise ValueError(f"The value flag, {flag} is invalid.")

        return torch.cat(dec_input_list, 0)

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
    parser.add_argument("--latent_size", default=256, type=int)

    ## the unused parameters
    parser.add_argument("--k", default=0.0025, type=float) 
    parser.add_argument("--x0", default=2500, type=int)
    parser.add_argument("--annealing_fn", default='logistic')

    # generation type
    parser.add_argument("--generation_type", default='gaussian', type=str)

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
            tokenizer=tokenizer
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

    # new a generator 
    ## generation config
    generator_config = {
            'num_beams': args.beam_size,
            'max_length': args.max_q_length,
            'do_sample': args.do_sample,
            'top_k': args.top_k,
    }
    generator_config['flags'] = args.flags

    generator = QuestionGenerator(
            model=model, 
            tokenizer=tokenizer, 
            n_samples=args.n_samples, 
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
            for i, output in output_dict.items():
                # for flag in args.flags:
                output.update(
                        {'prediction': predictions[i]}
                )
                f.write(json.dumps(output)+'\n')
    f.close()


    # transform to good read version
    from utils import transform_pred_to_good_read
    transform_pred_to_good_read(
            args.output_jsonl, args.output_jsonl.replace('jsonl', 'txt')
    )

