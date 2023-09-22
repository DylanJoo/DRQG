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
    # (tokenizer, prompt indices)
    tokenizer = AutoTokenizer.from_pretrained(hfmodel_args.tokenizer_name)
    prepare_prompt_idx(model_args, tokenizer)

    # Model
    from models.promptRelQG import SoftRelPromptFlanT5
    model = SoftRelPromptFlanT5.from_pretrained(
            hfmodel_args.model_name_or_path,
            model_args.instruction_prompt_idx,
            model_args.relevant_prompt_idx,
            model_args.irrelevant_prompt_idx
    )
    prompt_length = len(model_args.instruction_prompt_idx) 
    prompt_length += len(model_args.relevant_prompt_idx) 

    ## Freezing
    ### Prompt tuning (soft)
    if training_args.random_init:
        # pos from rand, neg from rand
        model.encoder.init_from_vocab(True, False)
    else:
        # pos from True, neg from False
        model.encoder.init_from_vocab()

    print('\n')
    for name, param in model.named_parameters():
        if 'prompt' in name:
            param.requires_grad = True
            print('param {} will be optimized.'.format(name))
        else:
            param.requires_grad = False
    print('\n')

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
            max_q_length=data_args.max_p_length,
            m_negatives=data_args.m_negative_per_example,
            m_positives=data_args.m_positive_per_example,
            prefix=model_args.baseline_prefix,
            scores=used_scores,
            prompt_length=prompt_length
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

    # Trainer
    from trainers import TrainerForRelQG
    trainer = TrainerForRelQG(
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

    return results

if __name__ == '__main__':
    main()
