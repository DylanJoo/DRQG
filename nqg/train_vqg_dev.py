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
    adaptive_pooling: Optional[str] = field(default=None)
    n_soft_prompts: int = field(default=1)
    latent_size: int = field(default=128)
    has_compressed_layer: bool = field(default=False)
    k: float = field(default=0.0025)
    x0: int = field(default=2500)
    annealing_fn: str = field(default='logistic')
    freeze_LM: bool = field(default=True)
    freeze_embeds: bool = field(default=True)
    freeze_a_layer: bool = field(default=True)
    freeze_cross_attn: bool = field(default=True)
    initialize_from_vocab: bool = field(default=True)
    used_prompt: str = field(default="<s>")
    n: int = field(default=1)
    n_side: int = field(default=None)
    add_attentive_pooler: bool = field(default=False)
    disable_dropout: bool = field(default=False)
    # random_masking_ratio: Optional[float] = field(default=None)
    add_classification_head: bool = field(default=False)

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
    m_negative_per_example: int = field(default=1)
    m_positive_per_example: int = field(default=1)

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
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        hfmodel_args, model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # [CONCERN] Deprecated? or any parser issue.
        hfmodel_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Config and Tokenizer
    config = AutoConfig.from_pretrained(hfmodel_args.config_name)

    if model_args.disable_dropout:
        config.activation_dropout=0
        config.attention_dropout=0
        config.classif_dropout=0
        config.classifier_dropout=0
        config.dropout=0

    tokenizer = AutoTokenizer.from_pretrained(hfmodel_args.tokenizer_name)

    # Model
    from models import T5VQG, BartVQGDEV
    MODELS = {"t5": T5VQG, 'bart': BartVQGDEV}
    for key in MODELS:
        if key in training_args.output_dir.lower():
            model_key = key

    # Model: Enc-Dec
    model_args.used_vocab_idx = tokenizer.encode(
            model_args.used_prompt,
            add_special_tokens=False
    )
    model_args.used_prompt = None
    print('Used vocab index initialization', model_args.used_vocab_idx)

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
    optimized_prefix = ['embed_tokens']

    if model_args.freeze_embeds is False:
        optimized_prefix.append('shared')
    if model_args.freeze_a_layer is False:
        optimized_prefix.append('encoder.layers.0') # first
    if model_args.freeze_cross_attn is False:
        optimized_prefix.append('encoder_attn')

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
    from datacollator import DataCollatorForVQG
    DATACOLLATORS = {
            "vl": DataCollatorForVQG
    }

    for key in DATACOLLATORS:
        if key in data_args.train_file.lower():
            datacollator_key = key

    data_collator = DATACOLLATORS[datacollator_key](
            tokenizer=tokenizer, 
            padding=True,
            return_tensors='pt',
            is_train=True,
            max_p_length=data_args.max_p_length,
            m_negatives=data_args.m_negative_per_example,
            m_positives=data_args.m_positive_per_example
    )

    # Data: dataset
    from data import msmarco, dragon, nils, redragon
    DATASETS = {
            "msmarco": msmarco, 'redragon': redragon, 
            'dragon': dragon, 'nils': nils
    }

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
    from trainers_dev import TrainerForVQG
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
