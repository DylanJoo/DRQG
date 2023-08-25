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
from arguments_new import *

import os
os.environ["WANDB_DISABLED"] = "false"

def prepare_prompt_idx(opt, tokenizer):
    get_tokenized_idx = lambda x: tokenizer.encode(x, add_special_tokens=False)

    if opt.instruct_prompt:
        opt.instruct_prompt_idx = get_tokenized_idx(opt.instruct_prompt)
        print('Used instruction prompt:', opt.instruct_prompt_idx)
    if opt.relevance_prompt:
        opt.relevance_prompt_idx = get_tokenized_idx(opt.relevance_prompt)
        print('Used relevance prompt:', opt.relevance_prompt_idx)

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
    from models import FlanT5
    model = FlanT5.from_pretrained(hfmodel_args.model_name_or_path)

    # model freezed
    for name, param in model.named_parameters():
        # tune the shared embeddings
        if training_args.prefix_tuning and ('shared' in name):
            param.requires_grad = True
            print('param {} will be optimized.'.format(name))
        else:
            param.requires_grad = False

    # Generation config
    generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config = generation_config

    # Data
    # Datacollator
    from data import DataCollatorForBaseline
    used_scores = list(range(0, 101, 101//10))
    used_scores = [s*0.01 for s in used_scores]
    data_collator = DataCollatorForBaseline(
            tokenizer=tokenizer, 
            max_p_length=data_args.max_p_length,
            max_q_length=data_args.max_p_length,
            m_negatives=data_args.m_negative_per_example,
            m_positives=data_args.m_positive_per_example,
            prefix=model_args.baseline_prefix,
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
                    test_size=99, 
                    train_size=min(n_examples-99, 400000), 
                    seed=1997
            )
        else:
            dataset['test'] = load_dataset('json', data_files=data_args.eval_file)['train']
    else:
        dataset['test'] = None

    # Trainer
    from trainers import TrainerForQG
    trainer = TrainerForQG(
            model=model, 
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator,
    )
    trainer.set_tokenizer(tokenizer)
    trainer.set_prefix(model_args.baseline_prefix)
    
    results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    return results

if __name__ == '__main__':
    main()
