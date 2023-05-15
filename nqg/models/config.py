from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any

@dataclass
class VQG_OLD_CONFIG:
    initialize_from_vocab: bool = field(default=True)
    pooling: Optional[str] = field(default='static')
    n_soft_prompts: Optional[int] = field(default=1)
    latent_size: Optional[int] = field(default=128)
    k: float = field(default=0.0025)
    x0: int = field(default=2500)
    annealing_fn: str = field(default='logistic')
    freeze_LM: bool = field(default=False)

@dataclass
class VQG_CONFIG:
    pooling: str = field(default='static')
    n_soft_prompts: int = field(default=1)
    latent_size: int = field(default=128)
    k: float = field(default=0.0025)
    x0: int = field(default=2500)
    annealing_fn: str = field(default='logistic')
    freeze_LM: bool = field(default=False)
    freeze_embeds: bool = field(default=False)
    initialize_from_vocab: bool = field(default=False)
    n: int = field(default=None)
    n_side: int = field(default=5)
