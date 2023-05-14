"""
TODO: tidy up imported modules
"""
# import torch
# from typing import Optional, Tuple, Union
# from transformers.models.t5.modeling_t5 import T5Stack
# from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
# from torch import nn
# from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
# from utils import kl_weight, kl_loss
# import copy

from transformers import T5ForConditionalGeneration
class T5PQG(T5ForConditionalGeneration):

    def set_tokenizer(self, tokenizer=None):
        self.tokenizer = tokenizer

from transformers import BartForConditionalGeneration
class BartPQG(BartForConditionalGeneration):

    def set_tokenizer(self, tokenizer=None):
        self.tokenizer = tokenizer
