import sys
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    GenerationConfig
)
from trainers import TrainerForT5
from datasets import load_dataset

import os
os.environ["WANDB_DISABLED"] = "false"

@dataclass
class OurHFModelArguments:
    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(default='t5-base')
    config_name: Optional[str] = field(default='t5-base')
    tokenizer_name: Optional[str] = field(default='t5-base')
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)

@dataclass
class OurModelArguments:
    pooling: str = field(default='static')
    n_soft_prompts: int = field(default=1)
    latent_size: int = field(default=128)
    k: float = field(default=0.0025)
    x0: int = field(default=2500)
    annealing_fn: str = field(default='logistic')
    freeze_t5: bool = field(default=False)

@dataclass
class OurDataArguments:
    # Huggingface's original arguments. 
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    # Customized arguments
    train_file: Optional[str] = field(default=None)
    eval_file: Optional[str] = field(default=None)
    max_p_length: int = field(default=256)
    max_q_length: int = field(default=16)
    # collection: Optional[str] = field(default=None)
    # p_centric_triplet: Optional[str] = field(default='triples.train.small.v1.sample.jsonl')

@dataclass
class OurTrainingArguments(TrainingArguments):
    # Huggingface's original arguments. 
    output_dir: str = field(default='./temp')
    seed: int = field(default=42)
    data_seed: int = field(default=None)
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    max_steps: int = field(default=10000)
    save_steps: int = field(default=5000)
    eval_steps: int = field(default=2500)
    evaluation_strategy: Optional[str] = field(default='no')
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    logging_dir: Optional[str] = field(default='./logs')
    resume_from_checkpoint: Optional[str] = field(default=None)
    save_total_limit: Optional[int] = field(default=5)
    learning_rate: Optional[float] = field(default=5e-5)
    lr_scheduler_type: Union[str] = field(default='linear')
    # Customized arguments
    remove_unused_columns: bool = field(default=False)

def main():

    # Parseing argument for huggingface packages
    parser = HfArgumentParser((OurHFModelArguments, OurModelArguments, OurDataArguments, OurTrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_datalcasses()
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        hfmodel_args, model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # [CONCERN] Deprecated? or any parser issue.
        hfmodel_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # additional config for models
    config = AutoConfig.from_pretrained(hfmodel_args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(hfmodel_args.tokenizer_name)

    from models import T5VQGDEV
    MODELS = {"t5vqg": T5VQGDEV}

    model_key = 't5vqg' # default
    for key in MODELS:
        if key in hfmodel_args.model_name_or_path.lower():
            model_key = key

    model = MODELS[model_key].from_pretrained(
            pretrained_model_name_or_path=hfmodel_args.model_name_or_path,
            config=config, 
            vae_config=model_args,
            tokenizer=tokenizer
    )
    
    ## add generation config
    generation_config = GenerationConfig.from_pretrained(hfmodel_args.config_name)
    generation_config.update(
        _from_model_config=False,
        num_beams=1,
        max_length=data_args.max_q_length
    )
    model.generation_config = generation_config

    ## data collator
    from datacollator import DataCollatorForT5Dev, DataCollatorForT5vL
    DATACOLLATORS = {"t5vqg": DataCollatorForT5Dev, "vl": DataCollatorForT5vL}

    datacollator_key = 't5vqg' # default
    for key in DATACOLLATORS:
        if key in data_args.train_file.lower():
            datacollator_key = key

    data_collator = DATACOLLATORS[datacollator_key](
            tokenizer=tokenizer, 
            padding=True,
            max_length=data_args.max_p_length,
            return_tensors='pt',
            is_train=True,
    )

    # freezing parameters
    optimized_prefix = ['hidden2', 'latent', 'soft', 'prompt']

    if model_args.freeze_t5:
        print('\nThe fine-tuned components:\n')
        for name, param in model.named_parameters():
            if any([p in name for p in optimized_prefix]):
                print('param {}: {}'.format(name, param.grad))
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Dataset
    from data import msmarco, dragon
    DATASETS = {"msmarco": msmarco, 'dragon': dragon}

    dataset_key = 'msmarco' # default
    for key in DATASETS:
        if key in data_args.train_file.lower():
            dataset_key = key

    dataset = DATASETS[dataset_key].passage_centric_dataset(data_args.train_file)

    if training_args.do_eval is True:
        if data_args.eval_file is None:
            dataset = dataset['train'].train_test_split(
                    test_size=99, train_size=400000
            )
        else:
            dataset['test'] = \
                    load_dataset('json', data_files=data_args.eval_file)['train']
    else:
        dataset['test'] = None

    # Trainer
    trainer = TrainerForT5(
            model=model, 
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator,
    )
    
    # ***** strat training *****
    results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
    )
    trainer.save_model()

    return results

if __name__ == '__main__':
    main()
