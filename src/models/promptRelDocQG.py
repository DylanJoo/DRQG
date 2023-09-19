import copy
import torch
import inspect
from torch import nn
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from models import FlanT5
from transformers import T5Tokenizer

class SoftRelPromptDocFlanT5(FlanT5):

    def __init__(self, config: T5Config, 
                 instruction_prompt_idx: Optional[List[int]] = None, 
                 relevance_prompt_idx: Optional[List[int]] = None,
                 nonrelevance_prompt_idx: Optional[List[int]] = None):

        super().__init__(config)
        print('Used instruction prompt:', instruction_prompt_idx)
        print('Used relevance prompt:', relevance_prompt_idx)
        print('Used nonrelevance prompt:', nonrelevance_prompt_idx)

        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = SoftRelPromptDocT5Stack(
                instruction_idx=instruction_prompt_idx,
                relevance_idx=relevance_prompt_idx,
                nonrelevance_idx=nonrelevance_prompt_idx,
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
        attention_mask = attention_mask[:,1+self.encoder.memory_indices[-1]:]
        return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                **kwargs
        )

class SoftRelPromptDocT5Stack(T5Stack):

    def __init__(self, 
                 instruction_idx=None, 
                 relevance_idx=None, 
                 nonrelevance_idx=None, 
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
        self.relevance_idx = torch.LongTensor(relevance_idx)
        self.positive_prompt = nn.Parameter(torch.rand(
            len(relevance_idx), embed_tokens.embedding_dim
        ))
        self.nonrelevance_idx = torch.LongTensor(nonrelevance_idx)
        self.negative_prompt = nn.Parameter(torch.rand(
            len(nonrelevance_idx), embed_tokens.embedding_dim
        ))

        # projection layer
        start = len(self.instruction_idx) # the index of 1st rel tok
        end = start + len(self.relevance_idx)  # the indeex of last rel tok
        self.memory_indices = list(range(start, end))
        self.doc_proj = nn.Linear(kwargs['config'].d_model, kwargs['config'].d_model)

    def init_from_vocab(self, positive=True, negative=True):
        self.instruction_prompt = nn.Parameter(
                self.wte(self.instruction_idx).clone().detach()
        )
        if positive:
            self.positive_prompt = nn.Parameter(
                    self.wte(self.relevance_idx).clone().detach()
            )
        if negative:
            self.negative_prompt = nn.Parameter(
                    self.wte(self.nonrelevance_idx).clone().detach()
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
            relevance_prompts = torch.matmul(
                    rel_scores.view(-1, 1), 
                    self.positive_prompt.view(1, -1)
            ) + torch.matmul(
                    (1-rel_scores).view(-1, 1), 
                    self.negative_prompt.view(1, -1)
            )
            relevance_prompts = relevance_prompts.view(B, -1, H)
            prompts += [relevance_prompts]

        inputs_embeds = torch.cat(prompts + [inputs_embeds], dim=1)
        outputs = super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=True,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
        )
        outputs['last_hidden_state'] = \
                self._reformulate(outputs['last_hidden_state'])

        # N Nh Le Le --> N Nh Le --> N Le
        # attention_outputs = []
        # for attention in outputs['attentions']:
        #     attention_outputs.append(attention.sum(-2).sum(1))
        # attention_outputs = torch.stack(attention_outputs).mean(0)
        # max_attn_tokens = torch.topk(attention_outputs[:, self.memory_indices[-1]:], dim=-1, k=10).indices
        # a = input_ids.clone().detach().cpu().numpy()
        # b = max_attn_tokens.clone().detach().cpu().numpy()
        # for i in range(4):
        #     print(self.tokenizer.decode(a[i, b[i, :]]))
        #     print(a[i, b[i, :]])

        return outputs

    def _reformulate(self, sequence_outputs):
        # type1: only use relevance prompt embeddings
        # sequence_outputs = sequence_outputs[:, self.memory_indices, :]

        # type2: only use relevance-aware passage embeddings
        sequence_outputs = sequence_outputs[:, 1+self.memory_indices[-1]:, :]

        return self.doc_proj(sequence_outputs)
