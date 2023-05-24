import torch
import inspect
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import BartForConditionalGeneration, BartConfig, BartModel
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.modeling_outputs import Seq2SeqLMOutput

from torch import nn
from torch.nn import CrossEntropyLoss

from utils import kl_weight, kl_loss
import copy
from .questiongenerator import BartQG
from .prompt import (
        SoftEmbedding, 
        SoftStaticEmbedding, 
        SoftAdaptiveEmbedding, 
        SoftEmbeddingWithPooler, 
)

PROMPT_EMBEDS = {
        'basic': SoftEmbedding,
        'static': SoftStaticEmbedding,
        'adaptive': SoftAdaptiveEmbedding,
        'attentive': SoftEmbeddingWithPooler,
        # 'residual': SoftResidualEmbedding
}

class BartVQG(BartQG):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"lm_head.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, config: BartConfig, vqg_config=None):
        super().__init__(config)

        # Reparameterized initialization
        self.n_soft_prompts = vqg_config.n_soft_prompts

        # soft prompting
        ## config
        kwargs = {
                'annealing_fn': vqg_config.annealing_fn, 
                'k': vqg_config.k, 'x0': vqg_config.x0,
                'has_compressed_layer': vqg_config.has_compressed_layer
        }
        ## pooler
        if vqg_config.add_attentive_pooler:
            pooler = self.get_pooler(config)
        else:
            pooler = None

        ## prompt layer
        self.enc_prompts = PROMPT_EMBEDS[vqg_config.pooling](
                wte=self.model.shared, 
                hidden_size=config.d_model,
                latent_size=vqg_config.latent_size,
                initialize_from_vocab=vqg_config.initialize_from_vocab,
                used_vocab_idx=vqg_config.used_vocab_idx,
                n_prompts=vqg_config.n_soft_prompts,
                pooler=pooler,
                adaptive_pooling=vqg_config.adaptive_pooling,
                **kwargs
        )
        self.model.encoder.set_input_embeddings(self.enc_prompts)
        self.model.decoder.set_input_embeddings(self.model.shared)
        # self.model.decoder.set_input_embeddings(self.dec_prompts)

        # set evaluation sample when inference temp results
        # attrs: (1) n_samples (2) name_samples
        self.set_n_eval_samples(n=vqg_config.n, n_side=vqg_config.n_side)
        self.enc_prompts.set_gaussian_range(self.name_samples)
        print(f'Prompt embedding set finished with \
                \n{self.n_samples} eval samples: {self.name_samples}.')

    def reparam_inputs(self, input_ids, attn_mask, steps=None, clf_labels=None):
        """
        Transform the inputs to reparemterized inputs.
        input_ids --> add variational prompts
        attn_mask --> add prompt-length attention
        """
        # input_ids --> input_embeds 
        inputs_embeds = self.model.encoder.embed_tokens(
                input_ids.view(-1, input_ids.size()[-1]),
                is_train=(steps is not None), 
                steps=steps,
                clf_labels=clf_labels
        ) 
        soft_attn_mask = torch.cat([
            torch.ones((attn_mask.size(0), self.n_soft_prompts)).to(attn_mask.device),
            attn_mask
        ], 1)

        return inputs_embeds, soft_attn_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        steps: Optional[int] = None,
        clf_labels: Optional[torch.LongTensor] = None,
        clf_scores: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        if encoder_outputs is None: # the first momnet when generating 
            inputs_embeds, attention_mask_new = self.reparam_inputs(
                    input_ids, attention_mask, steps, clf_labels
            )
        else:
            inputs_embeds = None
            attention_mask_new = attention_mask

        # standard enc-dec pipeline
        # TODO add classfication task head here
        return super().forward(
                input_ids=None, 
                attention_mask=attention_mask_new,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
        )

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        inputs_embeds, attention_mask_new = self.reparam_inputs(
                input_ids, attention_mask, steps=None
        )
        attention_mask_new = attention_mask_new.repeat(self.n_samples, 1)
        return super().generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask_new, 
                **kwargs
        )

