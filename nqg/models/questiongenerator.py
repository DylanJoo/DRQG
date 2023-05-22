from transformers import T5ForConditionalGeneration, BartForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.bart.modeling_bart import BartEncoderLayer

class T5QG(T5ForConditionalGeneration):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        if len(kwargs) != 0:
            print(f"{kwargs} would not be used in this model.")
        self.n_samples = 1

    def get_pooler(self, config=None):
        config.activation_dropout=0.1
        config.attention_dropout=0.1
        config.classif_dropout=0.1
        config.classifier_dropout=0
        config.dropout=0.1
        return T5Stack(config)

    def set_n_eval_samples(self, n=None, n_side=None):
        self.name_samples = list(range(n))
        self.n_samples = n

        if n_side is not None:
            self.name_samples = list(range(-n_side, n_side+1, 1))
            self.n_samples = 2*n_side+1

    def set_tokenizer(self, tokenizer=None):
        self.tokenizer = tokenizer

class BartQG(BartForConditionalGeneration):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        if len(kwargs) != 0:
            print(f"{kwargs} would not be used in this model.")
        self.n_samples = 1

    def get_pooler(self, config=None):
        return BartEncoderLayer(config)

    def set_tokenizer(self, tokenizer=None):
        self.tokenizer = tokenizer

    def set_n_eval_samples(self, n=None, n_side=None):
        self.n_samples = n

        if n_side is not None:
            self.name_samples = list(range(-n_side, n_side+1, 1))
            self.n_samples = 2*n_side+1
