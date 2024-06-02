import sys
import multiprocessing
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    HfArgumentParser,
    GenerationConfig
)
from datasets import load_dataset
from arguments import *

import os

def prepare_prompt_idx(opt, tokenizer):
    get_tokenized_idx = lambda x: tokenizer.encode(x, add_special_tokens=False)

    if opt.instruction_prompt:
        opt.instruction_prompt_idx = get_tokenized_idx(opt.instruction_prompt)
    if opt.relevant_prompt:
        opt.relevant_prompt_idx = get_tokenized_idx(opt.relevant_prompt)
    if opt.irrelevant_prompt:
        opt.irrelevant_prompt_idx = get_tokenized_idx(opt.irrelevant_prompt)

def main():
    # Parse argument for huggingface packages
    parser = HfArgumentParser(
            (HFModelArgs, ModelArgs, DataArgs, TrainingArgs)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        hfmodel_args, model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        hfmodel_args, model_args, data_args, training_args = \
                parser.parse_args_into_dataclasses()

    # Preparation 
    # (tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(hfmodel_args.tokenizer_name)
    prepare_prompt_idx(model_args, tokenizer)

    # Model
    ## [May 11] guess this is the oldest one.
    # from models import FlanT5
    # model = FlanT5.from_pretrained(hfmodel_args.model_name_or_path)
    from models.promptQGBase import SoftPromptFlanT5
    model = SoftPromptFlanT5.from_pretrained(
            hfmodel_args.model_name_or_path,
            model_args.instruction_prompt_idx
    )
    prompt_length = len(model_args.instruction_prompt_idx)
    model.encoder.init_from_vocab()

    print('\n')
    for name, param in model.named_parameters():
        if 'prompt' in name:
            param.requires_grad = True
            print('param {} will be optimized.'.format(name))
        else:
            param.requires_grad = False

    ## Generation config
    generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config = generation_config

    # Data
    # Datacollator
    from data import DataCollatorForPromptQG
    used_scores = list(range(0, 101, 101//10))
    used_scores = [s*0.01 for s in used_scores]
    data_collator = DataCollatorForPromptQG(
            tokenizer=tokenizer, 
            max_p_length=data_args.max_p_length,
            max_q_length=data_args.max_q_length,
            m_negatives=data_args.m_negative_per_example,
            m_positives=data_args.m_positive_per_example,
            prefix=model_args.baseline_prefix,
            prompt_length=prompt_length,
            scores=used_scores
    )

    # Data
    # Dataset
    from data import nils
    dataset = nils.passage_centric_dataset(data_args.train_file)
    n_examples = len(dataset['train'])

    if training_args.do_eval:
        if data_args.eval_file is None:
            dataset = dataset['train'].train_test_split(
                    test_size=1000, 
                    train_size=min(n_examples-1000, 400000), 
                    seed=1997
            )
        else:
            dataset['test'] = load_dataset('json', data_files=data_args.eval_file)['train']
    else:
        dataset['test'] = None

    print(n_examples)
    exit(0)
    # Trainer
    from trainers import TrainerBase
    trainer = TrainerBase(
            model=model, 
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator,
    )
    trainer.set_tokenizer(tokenizer)
    results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    # output configs
    model_args.dumps(training_args.output_dir+'/model_config.json')
    hfmodel_args.dumps(training_args.output_dir+'/hfmodel_config.json')
    data_args.dumps(training_args.output_dir+'/data_config.json')
    training_args.dumps(training_args.output_dir+'/train_config.json')


    return results

if __name__ == '__main__':
    main()
