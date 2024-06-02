import copy
import torch
import inspect
from torch import nn
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from models import FlanT5

class SoftRelPromptFlanT5(FlanT5):

    def __init__(self, config: T5Config, 
                 instruction_prompt_idx: Optional[List[int]] = None, 
                 relevant_prompt_idx: Optional[List[int]] = None,
                 irrelevant_prompt_idx: Optional[List[int]] = None, 
                 read_kwargs: Optional[Dict] = None):

        super().__init__(config)
        print('Used instruction prompt:', instruction_prompt_idx)
        print('Used relevant prompt:', relevant_prompt_idx)
        print('Used irrelevant prompt:', irrelevant_prompt_idx)
        self.prompt_length = (
                len(instruction_prompt_idx), len(relevant_prompt_idx)
        )

        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = SoftRelPromptT5Stack(
                instruction_idx=instruction_prompt_idx,
                relevant_idx=relevant_prompt_idx,
                irrelevant_idx=irrelevant_prompt_idx,
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

        # Read kwargs
        self.read_kwargs = read_kwargs

    # [TIPS]
    ## when generation, `input_ids` and `attention_mask` are not required
    ## since `encoder_outputs` has been seperately outputed.
    def forward(self, 
                input_ids=None,  
                attention_mask=None, 
                rel_scores=None, 
                encoder_outputs=None, 
                output_hidden_states=True,
                return_loss=True,
                **kwargs):

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    rel_scores=rel_scores,
                    **kwargs
            )

        # discard the prompts
        if self.read_kwargs['activate_prompt_attention'] is False:
            attention_mask[:, :sum(self.prompt_length)] = 0

        return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                output_hidden_states=output_hidden_states,
                **kwargs
        )

class SoftRelPromptT5Stack(T5Stack):

    def __init__(self, 
                 instruction_idx=None, 
                 relevant_idx=None, 
                 irrelevant_idx=None, 
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

        # relevant prompting
        self.relevant_idx = torch.LongTensor(relevant_idx)
        self.relevant_prompt = nn.Parameter(torch.rand(
            len(relevant_idx), embed_tokens.embedding_dim
        ))
        self.irrelevant_idx = torch.LongTensor(irrelevant_idx)
        self.irrelevant_prompt = nn.Parameter(torch.rand(
            len(irrelevant_idx), embed_tokens.embedding_dim
        ))

    def init_from_vocab(self, positive=True, negative=True):
        self.instruction_prompt = nn.Parameter(
                self.wte(self.instruction_idx).clone().detach()
        )
        if positive:
            self.relevant_prompt = nn.Parameter(
                    self.wte(self.relevant_idx).clone().detach()
            )
        if negative:
            self.irrelevant_prompt = nn.Parameter(
                    self.wte(self.irrelevant_idx).clone().detach()
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
        H = inputs_embeds.shape[2] 

        ## Expand customized prompts in front of `inputs_embeds` 
        prompts = []
        if self.instruction_idx is not None:
            # instruction_prompt: (N H) --> (B N H)
            prompts += [self.instruction_prompt.repeat(B, 1, 1)]

        if rel_scores is not None:
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
        outputs = super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
        )
        return outputs
