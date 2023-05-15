import sys
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import (
    AutoConfig,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Trainer,
    HfArgumentParser,
    GenerationConfig
)

import os
os.environ["WANDB_DISABLED"] = "false"

@dataclass
class OurHFModelArguments:
    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    use_auth_token: bool = field(default=False)

@dataclass
class OurDataArguments:
    # Huggingface's original arguments. 
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    # Customized arguments
    train_file: Optional[str] = field(default=None)
    # doc_query_pairs.train.jsonl
    eval_file: Optional[str] = field(default=None)
    max_p_length: int = field(default=256)
    max_q_length: int = field(default=16)

@dataclass
class OurTrainingArguments(Seq2SeqTrainingArguments):
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
    learning_rate: Union[float] = field(default=1e-5)
    lr_scheduler_type: Union[str] = field(default='linear')
    warmup_ratio: Union[float] = field(default=0.0)
    warmup_steps: Union[int] = field(default=0)
    # Customized arguments
    remove_unused_columns: bool = field(default=False)

def main():

    # Parseing argument for huggingface packages
    parser = HfArgumentParser(
            (OurHFModelArguments, OurDataArguments, OurTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        hfmodel_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        hfmodel_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Config and Tokenizer 
    config = AutoConfig.from_pretrained(hfmodel_args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(hfmodel_args.tokenizer_name)

    # Model: backbone and pretrained 
    from models import T5QG, BartQG
    MODELS = {"t5": T5QG, "bart": BartQG}
    for key in MODELS:
        if key in training_args.output_dir.lower():
            model_key = key

    model = MODELS[key].from_pretrained(
            hfmodel_args.model_name_or_path, config=config, 
    )
    model.set_tokenizer(tokenizer)
    
    # Model: generation config
    try:
        generation_config = GenerationConfig.from_pretrained(hfmodel_args.config_name)
    except:
        if 'bart' in hfmodel_args.config_name:
            generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config = generation_config

    # Data: collator/preprocessor
    from datacollator import DataCollatorBase
    data_collator = DataCollatorBase(
            tokenizer=tokenizer,
            max_p_length=data_args.max_p_length,
            max_q_length=data_args.max_q_length,
            is_eval=False
    )
    # Data: dataset
    from data import msmarco, dragon
    DATASETS = {"msmarco": msmarco, "dragon": dragon}
    for key in DATASETS:
        if key in data_args.train_file.lower():
            dataset_key = key

    dataset = DATASETS[dataset_key].passage_centric_dataset(data_args.train_file)

    from trainers import TrainerBase
    trainer = TrainerBase(
            model=model, 
            args=training_args,
            train_dataset=dataset['train'],
            data_collator=data_collator
    )
    
    # ***** strat training *****
    results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    return results

if __name__ == '__main__':
    main()
