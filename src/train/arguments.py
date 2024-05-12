import os
import json
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Union, List
from transformers import Seq2SeqTrainingArguments

class Configs:
    def dumps(self, path='temp.json'):
        temp_dict = dataclasses.asdict(self)
        with open(path, 'w') as f:
            f.write(json.dumps(temp_dict)+'\n')
        print('config {} has been saved at {}'.format(
            self.__class__.__name__, path
        ))

@dataclass
class HFModelArgs(Configs):
    model_name_or_path: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    use_auth_token: bool = field(default=False)

@dataclass
class ModelArgs(Configs):
    # disable_dropout: bool = field(default=False)
    add_classification_head: bool = field(default=False)

    # Baseline and hard prompt # {1} means passage
    baseline_prefix: Optional[str] = field(default='{0}')

    # Soft prompt 
    instruction_prompt: Optional[str] = field(default=None)
    instruction_prompt_idx: Optional[str] = field(default=None)
    ## Soft relevance prompt (single vector)
    pos_neg_prompt: Optional[str] = field(default=None)
    pos_neg_prompt_idx: Optional[str] = field(default=None)
    ## Soft relevance prompt (multiple vector)
    relevant_prompt: Optional[str] = field(default=None)
    relevant_prompt_idx: Optional[str] = field(default=None)
    irrelevant_prompt: Optional[str] = field(default=None)
    irrelevant_prompt_idx: Optional[str] = field(default=None)

    # Controller
    head_size: int = field(default=64)
    pooling: Optional[str] = field(default='mean')
    activation: Optional[str] = field(default='sigmoid')

    # variational
    latent_size: int = field(default=128)
    # has_compressed_layer: bool = field(default=False)
    # annealing_fn: str = field(default='cyclic')
    # n_total_iter: Optional[int] = field(default=10000)
    # n_cycle: Optional[int] = field(default=10)
    activate_prompt_attention: bool = field(default=True)

@dataclass
class DataArgs(Configs):
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    train_file: Optional[str] = field(default=None)
    eval_file: Optional[str] = field(default=None)
    max_p_length: int = field(default=256)
    max_q_length: int = field(default=16)
    m_negative_per_example: int = field(default=1)
    m_positive_per_example: int = field(default=1)
    random_corrupt_rate: Optional[float] = field(default=None)

@dataclass
class TrainingArgs(Configs, Seq2SeqTrainingArguments):
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
    overwrite_output_dir: bool = field(default=True)
    # Customized arguments
    remove_unused_columns: bool = field(default=False)
    random_init: bool = field(default=False)
    # Unliklihood
    enable_unlikelihood: bool = field(default=False)

    # Calibration (prob)
    # Calibration (BERTScore)
    enable_calibration: Optional[str] = field(default=None)
    calibration_margin_ngrams: Optional[List[str]] = field(default=None)
    gamma: Optional[float] = field(default=1.0)

    # In-batch encoder similarity
    enable_similarity_loss: Optional[str] = field(default=None)
    document_wise_contrastive: bool = field(default=False)
    relevance_wise_contrastive: bool = field(default=False)
    tau: Optional[float] = field(default=1.0)
    # Sampling
    sample_random: bool = field(default=False)
    sample_topk: int = field(default=1)

    # VAE KL regular
    enable_vae_loss: bool = field(default=False)
    k: float = field(default=0.0025)
    x0: int = field(default=2500)
    annealing_fn: str = field(default='logistic')

