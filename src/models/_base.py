from transformers import AutoModelForSeq2SeqLM

class FlanT5(AutoModelForSeq2SeqLM):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.n_samples = 1

    def set_tokenizer(self, tokenizer=None):
        self.tokenizer = tokenizer

