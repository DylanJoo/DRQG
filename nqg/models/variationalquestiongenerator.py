import torch
import inspect
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import BartForConditionalGeneration, BartConfig, BartModel
from transformers.models.bart.modeling_bart import BartClassificationHead
from transformers.modeling_outputs import Seq2SeqLMOutput

from torch import nn
from torch.nn import CrossEntropyLoss

from utils import kl_weight, kl_loss
import copy
from .questiongenerator import BartQG
from .prompt import SoftEmbedding, SoftAdaptiveEmbedding, SoftAttentiveEmbedding

PROMPT_EMBEDS = {
        'static': SoftEmbedding,
        'adaptive': SoftAdaptiveEmbedding,
        'attentive': SoftAttentiveEmbedding
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
        }
        ## pooler
        if vqg_config.add_attentive_pooler:
            pooler = self.get_pooler(config)
        else:
            pooler = None

        ## prompt layer
        self.prompts = PROMPT_EMBEDS[vqg_config.pooling](
                wte=self.model.shared, 
                hidden_size=config.d_model,
                latent_size=vqg_config.latent_size,
                n_prompts=vqg_config.n_soft_prompts,
                pooler=pooler,
                **kwargs
        )
        self.model.encoder.set_input_embeddings(self.prompts)
        self.model.decoder.set_input_embeddings(self.model.shared)

        # set evaluation sample when inference temp results
        # attrs: (1) n_samples (2) name_samples
        self.set_n_eval_samples(n=vqg_config.n, n_side=vqg_config.n_side)
        self.prompts.set_gaussian_range(self.name_samples)
        print(f'Prompt embedding set finished with \
                \n{self.n_samples} eval samples: {self.name_samples}.')

    def reparam_inputs(self, input_ids, attn_mask, steps=None):
        """
        Transform the inputs to reparemterized inputs.
        input_ids --> add variational prompts
        attn_mask --> add prompt-length attention
        """
        # input_ids --> input_embeds 
        inputs_embeds = self.model.encoder.embed_tokens(
                input_ids.view(-1, input_ids.size()[-1]),
                is_train=(steps is not None), 
                steps=steps
        ) 
        # attention_mask --> attention_mask 
        if inputs_embeds.size(0) % input_ids.size(0) != 0:
            N = inputs_embeds.size(0) // (attn_mask.size(0)//2)
            attn_mask_unit = attn_mask[:attn_mask.size(0)//2, :]
        else:
            N = inputs_embeds.size(0) // attn_mask.size(0)
            attn_mask_unit = attn_mask

        ## (row expanded) (col expanded)
        soft_attn_mask_unit = torch.cat([
            torch.ones((attn_mask_unit.size(0), self.n_soft_prompts)).to(attn_mask_unit.device),
            attn_mask_unit
        ], 1)

        return inputs_embeds, soft_attn_mask_unit.repeat(N, 1)

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
        **kwargs
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        if encoder_outputs is None: # the first momnet when generating 
            inputs_embeds, attention_mask_new = self.reparam_inputs(
                    input_ids, attention_mask, steps
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
                decoder_attention_mask=decoder_head_mask,
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
        return super().generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask_new, 
                **kwargs
        )

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, 
        inputs_tensor: torch.Tensor, 
        model_kwargs, 
        model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value 
                for argument, value in encoder_kwargs.items() 
                if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True

        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        device: torch.device = None,
    ) -> torch.LongTensor:
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            return model_kwargs.pop("decoder_input_ids")
        else:
            decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
            if device is None:
                device = self.device
            return torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id

