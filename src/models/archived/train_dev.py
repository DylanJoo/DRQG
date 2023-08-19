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
from datasets import load_dataset

import os
os.environ["WANDB_DISABLED"] = "false"

@dataclass
class OurHFModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    use_auth_token: bool = field(default=False)

@dataclass
class OurModelArguments:
    pooling: str = field(default='static')
    n_soft_prompts: int = field(default=1)
    latent_size: int = field(default=128)
    k: float = field(default=0.0025)
    x0: int = field(default=2500)
    annealing_fn: str = field(default='logistic')
    freeze_LM: bool = field(default=False)
    freeze_embeds: bool = field(default=False)
    initialize_from_vocab: bool = field(default=False)
    n: int = field(default=1)
    n_side: int = field(default=None)
    random_masking_ratio: Optional[float] = field(default=None)

@dataclass
class OurDataArguments:
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    # Customized arguments
    train_file: Optional[str] = field(default=None)
    eval_file: Optional[str] = field(default=None)
    max_p_length: int = field(default=256)
    max_q_length: int = field(default=16)
    m_samples_per_example: int = field(default=1)

@dataclass
class OurTrainingArguments(TrainingArguments):
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
    warmup_ratio: Union[float] = field(default=0.0)
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

    # Config and Tokenizer
    config = AutoConfig.from_pretrained(hfmodel_args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(hfmodel_args.tokenizer_name)

    # Model
    from models import T5VQG, BartVQG
    MODELS = {"t5": T5VQG, 'bart': BartVQG}

    for key in MODELS:
        if key in training_args.output_dir.lower():
            model_key = key

    model = MODELS[model_key].from_pretrained(
            pretrained_model_name_or_path=hfmodel_args.model_name_or_path,
            config=config, 
            vqg_config=model_args
    )
    model.set_tokenizer(tokenizer)
    
    # Model: generation config
    try:
        generation_config = GenerationConfig.from_pretrained(
                hfmodel_args.config_name,
                _from_model_config=False,
                num_beams=1,
                max_length=data_args.max_q_length
        )
    except:
        if 'bart' in hfmodel_args.config_name:
            generation_config = GenerationConfig.from_model_config(model.config)

    model.generation_config = generation_config

    # Model: freezing LM
    # [NOTE] OK-ish
    optimized_prefix = ['hidden2', 'latent', 'soft', 'prompt']
    # [NOTE] the better one
    optimized_prefix = ['hidden2', 'latent', 'soft', 'prompt', 'shared']

    if model_args.freeze_embeds is False:
        optimized_prefix.append('shared')

    if model_args.freeze_LM:
        print('\nThe fine-tuned components:\n')
        for name, param in model.named_parameters():
            if any([p in name for p in optimized_prefix]):
                print('param {}: {}'.format(name, param.grad))
                param.requires_grad = True
            else:
                param.requires_grad = False


    # Data: collator
    ### TODO Change the name `v0/v1` since the models have same setups
    from datacollator import DataCollatorForVQGSPT, DataCollatorForVQGDIV
    DATACOLLATORS = {
            "v0": DataCollatorForVQGSPT, 
            "v1": DataCollatorForVQGSPT, 
            "vl": DataCollatorForVQGDIV
    }

    for key in DATACOLLATORS:
        if key in data_args.train_file.lower():
            datacollator_key = key

    data_collator = DATACOLLATORS[datacollator_key](
            tokenizer=tokenizer, 
            padding=True,
            max_length=data_args.max_p_length,
            return_tensors='pt',
            is_train=True,
            m_samples_per_example=data_args.m_samples_per_example,
            random_masking_ratio=model_args.random_masking_ratio,
    )

    # Data: dataset
    from data import msmarco, dragon
    DATASETS = {"msmarco": msmarco, 'dragon': dragon}

    for key in DATASETS:
        if key in data_args.train_file.lower():
            dataset_key = key

    dataset = DATASETS[dataset_key].passage_centric_dataset(data_args.train_file)

    if training_args.do_eval is True:
        if data_args.eval_file is None:
            dataset = dataset['train'].train_test_split(
                    test_size=99, train_size=min(len(dataset['train'])-99, 400000)
            )
        else:
            dataset['test'] = \
                    load_dataset('json', data_files=data_args.eval_file)['train']
    else:
        dataset['test'] = None

    # Trainer
    from trainers import TrainerForVQG
    trainer = TrainerForVQG(
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
