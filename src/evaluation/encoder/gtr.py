import torch
import copy
from transformers import T5EncoderModel, T5Config
from transformers.models.t5.modeling_t5 import T5Stack
import torch.nn as nn
import torch.nn.functional as F

class GTREncoder(T5EncoderModel):
    """ Please use the huggingface checkpoint 'DylanJHJ/gtr-t5-base'
    """
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        self.linear = nn.Linear(config.d_model, config.d_model, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, **inputs):
        model_output = self.forward(**inputs)
        pooled_embeddings = self.mean_pooling(model_output, inputs['attention_mask'])
        encoded_embeddings = F.normalize(self.linear(pooled_embeddings), p=2, dim=1)
        return encoded_embeddings
