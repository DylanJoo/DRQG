from transformers import T5ForConditionalGeneration

class FlanT5(T5ForConditionalGeneration):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.n_samples = 1

    def forward(self, 
                steps=None, 
                rel_labels=None, 
                rel_scores=None, 
                passage=None, 
                **kwargs):
        return super().forward(**kwargs)

