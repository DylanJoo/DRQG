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
from peft import (
    get_peft_model, 
    PromptTuningConfig, 
    TaskType, 
    PromptTuningInit
)

import os
os.environ["WANDB_DISABLED"] = "false"

def prepare_prompt_idx(opt, tokenizer):
    get_tokenized_idx = lambda x: tokenizer.encode(x, add_special_tokens=False)

    if opt.pos_neg_prompt:
        opt.pos_neg_prompt_idx = get_tokenized_idx(opt.pos_neg_prompt)

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
    from models import RelPromptFlanT5
    model = RelPromptFlanT5.from_pretrained(
            hfmodel_args.model_name_or_path,
            pos_neg_prompt_idx=model_args.pos_neg_prompt_idx
    )
    model.encoder.init_from_vocab()

    # Peft Config
    peft_config = PromptTuningConfig(
            peft_type="PROMPT_TUNING",
            task_type=TaskType.SEQ_2_SEQ_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=model_args.n_instruction_prompt,
            prompt_tuning_init_text=f"{model_args.instruction_prompt}",
            inference_mode=False,
            tokenizer_name_or_path=hfmodel_args.model_name_or_path,
    )
    # Peft Model
    model = get_peft_model(model, peft_config)
    for name, param in model.named_parameters():
        if ('encoder' in name) and ('prompt' in name):
            print(param.requires_grad)
            param.requires_grad = True
        if param.requires_grad:
            print(name)

    model.print_trainable_parameters()
    prompt_length = 1 if model_args.pos_neg_prompt is not None else 0

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
                    test_size=99, 
                    train_size=min(n_examples-99, 400000), 
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
