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
os.environ["WANDB_DISABLED"] = "false"


def main():
    # Parseing argument for huggingface packages
    parser = HfArgumentParser(
            (HFModelArgs, ModelArgs, DataArgs, TrainingArgs)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        hfmodel_args, model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # [CONCERN] Deprecated? or any parser issue.
        hfmodel_args, model_args, data_args, training_args = \
                parser.parse_args_into_dataclasses()

    # Config and Tokenizer
    config = AutoConfig.from_pretrained(hfmodel_args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(hfmodel_args.tokenizer_name)

    # Model
    # [Controller before Encoder]
    if model_args.prompts is not None:
        model_args.prompts_idx = tokenizer.encode(
                model_args.prompts, add_special_tokens=False
        )
        print('Used prompt index:', model_args.prompts_idx)

    if model_args.label_prompts is not None:
        model_args.label_prompts_idx = tokenizer.encode(
                model_args.label_prompts, add_special_tokens=False
        )
        print('Used label prompt index:', model_args.label_prompts_idx)

    # [Controller between Encoder Decoder]
    if (model_args.pos_anchors is not None) and (model_args.neg_anchors is not None):
        model_args.pos_anchors_idx = tokenizer.encode(
                model_args.pos_anchors, add_special_tokens=False
        )
        model_args.neg_anchors_idx = tokenizer.encode(
                model_args.neg_anchors, add_special_tokens=False
        )
        print('Used pos/neg anchors index:', \
                model_args.pos_anchors_idx, model_args.neg_anchors_idx)

    from models import DocRelBartQG, RelBartQG
    MODELS = {'bart-condqg': DocRelBartQG, 'bart-relqg': RelBartQG}
    for key in MODELS:
        if key in training_args.output_dir.lower():
            model_key = key

    model = MODELS[model_key].from_pretrained(
            pretrained_model_name_or_path=hfmodel_args.model_name_or_path,
            config=config, 
            cvqg_config=model_args,
            batch_size=training_args.per_device_train_batch_size,
            aggregate=('mean' in hfmodel_args.model_name_or_path)
    )
    model.set_tokenizer(tokenizer)
    if training_args.set_embeddings:
        model.controller.set_embeddings()
    
    # [generation config]
    generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config = generation_config

    # Model: freezing LM
    optimized_prefix = ['controller', 'reformulator', 'adapter', 'vae', 'classification']
    freezed_prefix = []

    if model_args.freeze_embeddings:
        freezed_prefix += ['shared.weight']
    if model_args.freeze_encoder:
        freezed_prefix += ['model.encoder']
    if model_args.freeze_decoder:
        freezed_prefix += ['model.decoder']

    for name, param in model.named_parameters():
        if any([p in name for p in optimized_prefix]):
            # print('param {} will be optimized.'.format(name))
            param.requires_grad = True
        elif any([p in name for p in freezed_prefix]):
            # print('param {} wont be optimized.'.format(name))
            param.requires_grad = False
        else:
            print('param {} will be optimized.'.format(name))
            param.requires_grad = True


    # Data: collator
    from datacollator import DataCollatorForVQG
    data_collator = DataCollatorForVQG(
            tokenizer=tokenizer, 
            padding=True,
            return_tensors='pt',
            is_train=True,
            max_p_length=data_args.max_p_length,
            m_negatives=data_args.m_negative_per_example,
            m_positives=data_args.m_positive_per_example
    )

    # Data: dataset
    from data import msmarco, dragon, nils
    DATASETS = {"msmarco": msmarco, 'dragon': dragon, 'nils': nils}

    for key in DATASETS:
        if key in data_args.train_file.lower():
            dataset_key = key
    dataset = DATASETS[dataset_key].passage_centric_dataset(data_args.train_file)

    if training_args.do_eval is True:
        if data_args.eval_file is None:
            dataset = dataset['train'].train_test_split(
                    test_size=99, train_size=min(len(dataset['train'])-99, 400000), 
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
    
    # ***** strat training *****
    results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
    )
    trainer.save_model()

    return results

if __name__ == '__main__':
    main()