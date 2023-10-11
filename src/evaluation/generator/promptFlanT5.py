import copy
import torch
from torch import nn
from typing import List, Optional
from transformers import T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import T5Stack

class SoftPromptFlanT5(T5ForConditionalGeneration):

    def __init__(self, config: T5Config, 
                 num_instruction_prompt_idx: Optional[int] = None, 
                 num_relevance_prompt_idx: Optional[int] = None,
        ):

        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.prompt_length = (num_instruction_prompt_idx, num_relevance_prompt_idx)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = SoftPromptT5Stack(
                num_instruction_idx=num_instruction_prompt_idx,
                num_relevance_idx=num_relevance_prompt_idx,
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

    def forward(self, 
                input_ids=None,  
                attention_mask=None, 
                rel_scores=None, 
                encoder_outputs=None, 
                return_loss=True,
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
                 num_instruction_idx=None, 
                 num_relevance_idx=None, 
                 embed_tokens=None, 
                 **kwargs):
        super().__init__(**kwargs)

        self.wte = embed_tokens
        self.instruction_prompt = nn.Parameter(torch.rand(
            num_instruction_idx, embed_tokens.embedding_dim
        ))
        self.relevant_prompt = nn.Parameter(torch.rand(
            num_relevance_idx, embed_tokens.embedding_dim
        ))
        self.irrelevant_prompt = nn.Parameter(torch.rand(
            num_relevance_idx, embed_tokens.embedding_dim
        ))

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
        H = inputs_embeds.shape[2] 

        ## Expand customized prompts in front of `inputs_embeds` 
        prompts = []
        prompts += [self.instruction_prompt.repeat(B, 1, 1)]

        relevant_prompts = torch.matmul(
                rel_scores.view(-1, 1), 
                self.relevant_prompt.view(1, -1)
        ) + torch.matmul(
                (1-rel_scores).view(-1, 1), 
                self.irrelevant_prompt.view(1, -1)
        )
        relevant_prompts = relevant_prompts.view(B, -1, H)
        prompts += [relevant_prompts]

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
