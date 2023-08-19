# from .vqg_old_duodist import T5VQG as T5VQGV0
# from .vqg_old_monodist import T5VQG as T5VQGV1
# from .vqg_t5 import T5VQG as T5VQG
# move this into variationalquestiongenerator

from .questiongenerator import T5QG, BartQG
from .config import VQG_CONFIG, VQG_OLD_CONFIG

from .conditionalQG import DocRelBartQG
from .relevanceQG import RelBartQG
from .memoryQG import MemBartQG
