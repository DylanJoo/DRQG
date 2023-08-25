# [TODO] probably leave only hard prompt in promptQG
import copy
import torch
from torch import nn
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from models import FlanT5

class SoftPromptFlanT5(FlanT5):

    def __init__(self, config: T5Config, 
                 enc_prompt_idx: Optional[List[int]] = None):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = SoftPromptT5Stack(
                prompt_idx=enc_prompt_idx, 
                embed_tokens=self.shared,
                config=encoder_config, 
        )

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

class SoftPromptT5Stack(T5Stack):

    def __init__(self, prompt_idx=None, embed_tokens=None, **kwargs):
        super().__init__(**kwargs)
        self.prompt_idx = torch.LongTensor(prompt_idx)
        self.wte = embed_tokens
        self.prompt_embed = nn.Parameter(torch.rand(
            len(prompt_idx), embed_tokens.embedding_dim
        ))

    def init_from_vocab(self):
        self.prompt_embed = nn.Parameter(
                self.wte(self.prompt_idx).clone().detach()
        )

    def forward(self, 
                input_ids=None, 
                attention_mask=None, 
                inputs_embeds=None, 
                **kwargs):

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        B = inputs_embeds.shape[0]

        ## Expand customized prompts in front of `inputs_embeds` 
        prompts = self.prompt_embed.repeat(B, 1, 1)
        inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        ## and the attention_mask as well
        # attention_mask = self._expand(attention_mask, prompts.shape[1])

        return super().forward(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
        )
