from .vqg_old_duodist import T5VQG as T5VQGV0
from .vqg_old_monodist import T5VQG as T5VQGV1
from .variationalquestiongenerator import BartVQG as BartVQG
from .questiongenerator import T5QG, BartQG
from .config import VAE_CONFIG

# move this into variationalquestiongenerator
from .vqg_t5 import T5VQG as T5VQG
