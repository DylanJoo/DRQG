import sys
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    GenerationConfig
)
from trainers import TrainerForT5
from datacollator import DataCollatorForT5VQG
import msmarco 

import os
os.environ["WANDB_DISABLED"] = "false"

from models import T5VQGV0, T5VQGV1, T5PQG
MODELS = {'vqgv0': T5VQGV0, 'vqgv1': T5VQGV1, 'pqg': T5PQG}

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
    latent_size: int = field(default=128)
    k: float = field(default=0.0025)
    x0: int = field(default=2500)
    annealing_fn: str = field(default='logistic')

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
    max_length: int = field(default=5)
    collection: Optional[str] = field(default=None)
    queries: Optional[str] = field(default=None)
    qrels: Optional[str] = field(default=None)
    p_centric_triplet: Optional[str] = field(default='triples.train.small.v0.sample.jsonl')

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
    # Customized arguments
    remove_unused_columns: bool = field(default=False)
    debug_mode: int = field(default=None)

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
    for key in MODELS:
        if key in training_args.output_dir:
            model = MODELS[key].from_pretrained(
                    pretrained_model_name_or_path=hfmodel_args.model_name_or_path,
                    config=config, 
                    vae_config=model_args,
                    tokenizer=tokenizer,
                    debug=training_args.debug_mode
            )
    
    ## add generation config
    generation_config = GenerationConfig.from_pretrained(hfmodel_args.config_name)
    generation_config.update(
        _from_model_config=False,
        num_beams=5,
        max_length=16
    )
    model.generation_config = generation_config

    ## data collator
    data_collator = DataCollatorForT5VQG(
            tokenizer=tokenizer, 
            padding=True,
            max_length=data_args.max_length,
            return_tensors='pt',
            is_train=True
    )

    # Trainer
    dataset = msmarco.passage_centric_triplet_dataset(data_args)
    N = len(dataset['train'])
    if training_args.do_eval and data_args.eval_file is None:
        # split 0.1% for evaluation
        dataset = dataset['train'].train_test_split(test_size=0.001)
    else:
        dataset['test'] = load_dataset('json', data_files=data_args.eval_file)['train']


    trainer = TrainerForT5(
            model=model, 
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator
    )
    
    # ***** strat training *****
    results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    return results

if __name__ == '__main__':
    main()