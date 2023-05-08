from .pqg import T5PQG
from .vqg_double_dist import T5VQG as T5VQGV0
from .vqg_single_dist import T5VQG as T5VQGV1
from .vqg_single_softprompt import T5VQG as T5VQGDEV
from .vqg_single_softprompt import T5VQG as T5VQGSPT

from dataclasses import dataclass, field
@dataclass
class VAE_CONFIG:
    n_soft_prompts: int = field(default=1)
    latent_size: int = field(default=128)
    k: float = field(default=0.0025)
    x0: int = field(default=2500)
    annealing_fn: str = field(default='logistic')
