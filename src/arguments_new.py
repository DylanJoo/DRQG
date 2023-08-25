from dataclasses import dataclass, field
from typing import Optional, Union, List
from transformers import Seq2SeqTrainingArguments

@dataclass
class HFModelArgs:
    model_name_or_path: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    use_auth_token: bool = field(default=False)

@dataclass
class ModelArgs:
    # disable_dropout: bool = field(default=False)
    add_classification_head: bool = field(default=False)
    baseline_prefix: Optional[str] = field(default='')

    # Controller
    head_size: int = field(default=64)
    pooling: Optional[str] = field(default='mean')
    activation: Optional[str] = field(default='sigmoid')

    # EarlyCtrlQG
    instruct_prompt: Optional[bool] = field(default=False)
    instruct_prompt_idx: Optional[str] = field(default=None)
    relevance_prompt: Optional[bool] = field(default=False)
    relevance_prompt_idx: Optional[str] = field(default=None)

    # EarlyCtrlQG
    # pos_anchors : Optional[str] = field(default=None)
    # neg_anchors : Optional[str] = field(default=None)
    # pos_anchors_idx = None
    # neg_anchors_idx = None

    # variational
    # latent_size: int = field(default=128)
    # has_compressed_layer: bool = field(default=False)
    # annealing_fn: str = field(default='cyclic')
    # n_total_iter: Optional[int] = field(default=10000)
    # n_cycle: Optional[int] = field(default=10)

    # freeze layers
    # freeze_encoder: Optional[bool] = field(default=True)
    # freeze_decoder: Optional[bool] = field(default=True)
    # freeze_embeddings: Optional[bool] = field(default=True)

@dataclass
class DataArgs:
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    train_file: Optional[str] = field(default=None)
    eval_file: Optional[str] = field(default=None)
    max_p_length: int = field(default=256)
    max_q_length: int = field(default=16)
    m_negative_per_example: int = field(default=1)
    m_positive_per_example: int = field(default=1)

@dataclass
class TrainingArgs(Seq2SeqTrainingArguments):
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
    set_embeddings: bool = field(default=True)
    prefix_tuning: bool = field(default=False)
