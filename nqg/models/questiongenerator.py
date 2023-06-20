from transformers import T5ForConditionalGeneration, BartForConditionalGeneration

class T5QG(T5ForConditionalGeneration):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        if len(kwargs) != 0:
            print(f"{kwargs} would not be used in this model.")
        self.n_samples = 1

    def set_tokenizer(self, tokenizer=None):
        self.tokenizer = tokenizer

class BartQG(BartForConditionalGeneration):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        if len(kwargs) != 0:
            print(f"{kwargs} would not be used in this model.")
        self.n_samples = 1

    def set_tokenizer(self, tokenizer=None):
        self.tokenizer = tokenizer
