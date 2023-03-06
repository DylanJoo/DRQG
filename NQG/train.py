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
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset, DatasetDict, concatenate_datasets
from models import T5VAEForConditionalGeneration
from trainers import T5VAETrainer
import msmarco 

import os
os.environ["WANDB_DISABLED"] = "true"

@dataclass
class OurModelArguments:
    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(default='t5-small')
    config_name: Optional[str] = field(default='t5-small')
    tokenizer_name: Optional[str] = field(default='t5-small')
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    # Customized arguments
    vae_latent_size: int = field(default=128)
    vae_k: float = field(default=0.0025)
    vae_x0: int = field(default=2500)
    vae_annealing_fn: str = field(default='logistic')

@dataclass
class OurDataArguments:
    # Huggingface's original arguments. 
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    # Customized arguments
    train_file: Optional[str] = field(default=None)
    max_length: int = field(default=256)
    triplet: Optional[str] = field(default=None)
    collection: Optional[str] = field(default=None)
    queries: Optional[str] = field(default=None)
    qrels: Optional[str] = field(default=None)

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
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    logging_dir: Optional[str] = field(default='./logs')
    resume_from_checkpoint: Optional[str] = field(default=None)
    # Customized arguments
    remove_unused_columns: bool = field(default=False)

def main():

    # Parseing argument for huggingface packages
    parser = HfArgumentParser((OurModelArguments, OurDataArguments, OurTrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_datalcasses()
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # [CONCERN] Deprecated? or any parser issue.
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # additional config for models
    config = AutoConfig.from_pretrained(model_args.config_name)
    config_kwargs = {
            "latent_size": model_args.vae_latent_size, 
            "k": model_args.vae_k,
            "x0": model_args.vae_x0,
            "annealing_fn": model_args.vae_annealing_fn,
            "output_hidden_states": True,
    }
    config.update(config_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    model = T5VAEForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=config,
            tokenizer=tokenizer
    )

    ## data collator
    data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, 
            padding=True,
            return_tensors='pt'
    )

    # Trainer
    train_dataset=msmarco.triplet_dataset(data_args)

    trainer = T5VAETrainer(
            model=model, 
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator
    )
    
    # ***** strat training *****
    results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    return results

if __name__ == '__main__':
    main()
