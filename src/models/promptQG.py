# [TODO] probably leave only hard prompt in promptQG
import copy
import torch
from torch import nn
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from models import FlanT5

class PrefixFlanT5(FlanT5):

    def __init__(self, config: T5Config, 
                 enc_prompt_idx: Optional[List[int]] = None):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = PromptT5Stack(
                prompt_idx=enc_prompt_idx, 
                config=encoder_config, 
                embed_tokens=self.shared
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

    # def forward(self, 
    #             input_ids, 
    #             attention_mask, 
    #             hard_prompt_ids=None, 
    #             encoder_outputs=None, 
    #             steps=None, 
    #             **kwargs):
    #
    #     if encoder_outputs is None:
    #         encoder_outputs = self.encoder(
    #                 input_ids=input_ids, 
    #                 attention_mask=attention_mask, 
    #                 hard_prompt_ids=hard_prompt_ids
    #         )
    #         attention_mask = self.encoder._expand(
    #                 attention_mask, hard_prompt_ids.shape[1]
    #         )
    #
    #     return super().forward(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             encoder_outputs=encoder_outputs,
    #             **kwargs
    #     )


class PromptT5Stack(T5Stack):

    def __init__(self, prompt_idx=None, **kwargs):
        super().__init__(**kwargs)
        self.prompt_idx = prompt_idx
        self.prompt_embed = None

    def init_embeddings(self):
        self.prompt_embed = self.embed_tokens.weight[\
                self.prompt_idx].clone().detach()

    def _expand(self, mask, length):
        additional_mask = torch.ones((mask.size(0), length), device=self.device)
        return torch.cat([additional_mask, mask], -1)

    def forward(self, 
                input_ids=None, 
                attention_mask=None, 
                inputs_embeds=None, 
                soft_prompt=False, 
                **kwargs):

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        B = inputs_embeds.shape[0]

        if soft_prompt:
            ## Expand customized prompts (length is self.expanded_length)
            ## in from of `inputs_embeds` 
            prompts = self.prompt_embed.repeat(B, 1, 1)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
            ## and the attention_mask as well
            attention_mask = self._expand(attention_mask, prompts.shape[1])

        return super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
        )
