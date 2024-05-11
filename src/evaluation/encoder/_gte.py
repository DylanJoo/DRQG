from transformers import BertModel
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

class GTEEncoder(BertModel):
    """ Please use the huggingface checkpoint 'thenlper/gte-base'
    """
    @staticmethod
    def mean_pooling(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(self, **inputs):
        model_output = self.forward(**inputs)
        pooled_embeddings = self.mean_pooling(model_output.last_hidden_state, inputs['attention_mask'])
        encoded_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
        return encoded_embeddings

