iimport copy
import torch
import inspect
from torch import nn
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from models import FlanT5, SoftRelPromptT5Stack

class SoftRelPromptFlanT5(FlanT5):

    def __init__(self, config: T5Config, 
                 instruction_prompt_idx: Optional[List[int]] = None, 
                 relevant_prompt_idx: Optional[List[int]] = None,
                 irrelevant_prompt_idx: Optional[List[int]] = None, 
                 latent_size: Optional[int] = 768):

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
                config=encoder_config
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
                return_loss=True,
                steps=None,
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

