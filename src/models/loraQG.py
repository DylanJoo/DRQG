"""
Make it into lora modes.
"""
import copy
import torch
import inspect
from torch import nn
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from models import FlanT5

class LoraFlanT5(FlanT5):

    def __init__(self, config: T5Config, 
                 instruction_prompt_idx: Optional[List[int]] = None, 
                 relevance_prompt_idx: Optional[List[int]] = None):

        super().__init__(config)
        print('Used instruction prompt:', instruction_prompt_idx)
        print('Used relevance prompt:', relevance_prompt_idx)

        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = SoftPromptT5Stack(
                instruction_idx=instruction_prompt_idx,
                relevance_idx=relevance_prompt_idx,
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

    # [TIPS]
    ## when generation, `input_ids` and `attention_mask` are not required
    ## since `encoder_outputs` has been seperately outputed.
    def forward(self, 
                input_ids=None,  
                attention_mask=None, 
                rel_scores=None, 
                encoder_outputs=None, 
                **kwargs):

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    rel_scores=rel_scores,
                    **kwargs
            )
        return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                **kwargs
        )

class SoftPromptT5Stack(T5Stack):

    def __init__(self, 
                 instruction_idx=None, 
                 relevance_idx=None, 
                 embed_tokens=None, 
                 **kwargs):
        super().__init__(**kwargs)

        self.wte = embed_tokens

        # instruction prompting
        if instruction_idx:
            self.instruction_idx = torch.LongTensor(instruction_idx)
            self.instruction_prompt = nn.Parameter(torch.rand(
                len(instruction_idx), embed_tokens.embedding_dim
            ))
        else:
            self.instruction_idx = None

        # relevance prompting
        if relevance_idx:
            self.relevance_idx = torch.LongTensor(relevance_idx)
            self.relevance_prompt = nn.Parameter(torch.rand(
                len(relevance_idx), embed_tokens.embedding_dim
            ))
        else:
            self.relevance_idx = None

    def init_from_vocab(self):
        if self.instruction_idx is not None:
            self.instruction_prompt = nn.Parameter(
                    self.wte(self.instruction_idx).clone().detach()
            )
        if self.relevance_idx is not None:
            self.relevance_prompt = nn.Parameter(
                    self.wte(self.relevance_idx).clone().detach()
            )

    def forward(self, 
                input_ids=None,
                attention_mask=None,
                inputs_embeds=None,
                rel_scores=None,
                head_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        B = inputs_embeds.shape[0]

        ## Expand customized prompts in front of `inputs_embeds` 
        prompts = []
        if self.instruction_idx is not None:
            # instruction_prompt: (N H) --> (B N H)
            prompts += [self.instruction_prompt.repeat(B, 1, 1)]

        if rel_scores is not None:
            # reshape: rel_score (B) --> (B 2)
            # concat: (2 H) --> (B 1 H)
            relevance_prompt = torch.matmul(
                    torch.cat([1-rel_scores, rel_scores], -1).view(2, -1).T,
                    self.relevance_prompt
            ).unsqueeze(1)
            prompts += [relevance_prompt]

        inputs_embeds = torch.cat(prompts + [inputs_embeds], dim=1)
        return super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
        )
