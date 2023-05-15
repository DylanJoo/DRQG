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
class T5QG(T5ForConditionalGeneration):

    def set_n_eval_samples(self, n=None, n_side=None):
        self.name_samples = list(range(n))
        self.n_samples = n

        if n_side is not None:
            self.name_samples = list(range(-n_side, n_side+1, 1))
            self.n_samples = 2*n_side+1

    def set_tokenizer(self, tokenizer=None):
        self.tokenizer = tokenizer

from transformers import BartForConditionalGeneration
class BartQG(BartForConditionalGeneration):

    def set_tokenizer(self, tokenizer=None):
        self.tokenizer = tokenizer

    def set_n_eval_samples(self, n=None, n_side=None):
        self.n_samples = n

        if n_side is not None:
            self.name_samples = list(range(-n_side, n_side+1, 1))
            self.n_samples = 2*n_side+1
