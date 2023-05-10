from .pqg import T5PQG
from .vqg_double_dist import T5VQG as T5VQGV0
from .vqg_single_dist import T5VQG as T5VQGV1
from .vqg_single_softprompt import T5VQG as T5VQGSPT
from .config import VAE_CONFIG

# development
from .model_dev import BartVQG as BartVQGSPT
