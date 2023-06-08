from typing import List, Optional, Tuple, Union, Dict, Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BartConfig 
from transformers.models.bart.modeling_bart import (
        BartModel,
        shift_tokens_right, 
        BartClassificationHead
)

from .outputs import Seq2SeqCVQGOutput
from .questiongenerator import BartQG
from .module import InstanceWisePrompt, RelevancePrompt, EncDecCVAE

from utils import kl_weight, kl_loss
import copy

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
        # self.n_soft_prompts = vqg_config.n_soft_prompts

        self.model = BartModel(config)

        # [generation] 
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)   

        # [classification]
        if cvqg_config.add_classification_head:
            self.classification_head = BartClassificationHead(
                config.d_model, config.d_model, 1,
                config.classifier_dropout,
            )
        else:
            self.classification_head = None

        self.post_init()

        # [conditional vae]
        kld_kwargs = {'annealing_fn': cvqg_Config.annealing_fn}
        if kld_kwargs['annealing_fn'] != 'cyclic':
            kld_kwargs.update({
                'k': cvqg_config.k, 'x0': cvqg_config.x0
            })
        else:
            kld_kwargs.update({
                'total_iter': cvqg_config.total_iter, 
                'n_cycle': cvqg_config.n_cycle
            }) 

        # [prompt] 
        self.prompts = DocRelPrompt(
                wte=self.model.shared, 
                hidden_size=config.d_model,
                init_idx=cvqg_config.used_prompt_idx,
                lbl_init_idx=cvqg_config.used_label_idx,
        )

        # [condition]
        self.encdec_cvae = EncDecCVAE(
                wte=self.model.shared,
                hidden_size=config.d_model,
                latent_size=cvqg_config.latent_size,
                prefix_length=len(cvqg_config.used_label_idx),
                has_compressed_layer=cvqg_config.has_compressed_layer,
        )

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
    ) -> Union[Tuple, Seq2SeqCVQGOutput]:

        # [parameters]
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None \
                    else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # [decoder inputs] 
        if labels is not None:
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # [Bart encoding]
        if encoder_outputs is None:
            ## [Encoder prompt wrapper]
            inputs_embeds = self.prompts(clf_scores, input_ids)
            attention_mask = self.prompts.expand(attention_mask)

            ## [Bart encoder]
            encoder_outputs = self.model.encoder(
                input_ids=None,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # wrapper for BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # standard enc-dec pipeline
        decoder_inputs_embeds = self.model.decoder.embed_tokens(decoder_input_ids)
        decoder_outputs = super().forward(
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
                output_hidden_states=True,
                return_dict=True,
                **kwargs
        )
        lm_logits = self.lm_head(decoder_outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        mased_lm_loss, reparam_loss = 0, 0
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                    lm_logits.view(-1, self.config.vocab_size), 
                    labels.view(-1)
            )

            if self.classification_head is not None:
                hidden_states = decoder_outputs.last_hidden_states
                decoder_input_ids = shift_tokens_right_modified(
                        labels, 
                        self.config.pad_token_id, 
                        self.config.eos_token_id
                )
                eos_mask = decoder_input_ids.eq(self.config.eos_token_id).to(hidden_states.device)
                sentence_representation = hidden_states[eos_mask, :]
                clf_logits = self.classification_head(sentence_representation)

        return Seq2SeqCVQGOutput(
                loss=masked_lm_loss, 
                reparam_loss=reparam_loss,
                logits=lm_logits, 
                clf_logits=clf_logits,
                past_key_values=outputs.past_key_values, 
                decoder_hidden_states=outputs.decoder_hidden_states, 
                decoder_attentions=outputs.decoder_attentions, 
                cross_attentions=outputs.cross_attentions, 
                encoder_last_hidden_state=outputs.encoder_last_hidden_state, 
                encoder_hidden_states=outputs.encoder_hidden_states, 
                encoder_attentions=outputs.encoder_attentions,
        )

def shift_tokens_right_modified(
        input_ids: torch.Tensor, 
        pad_token_id: int, 
        eos_token_id: int
    ):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = pad_token_id 

    # sometimes the last token is eos; thus, replace the last one by eos then.
    shifted_input_ids[input_ids[:, -1].eq(eos_token_id), -1] = eos_token_id
    # since we focus the eos token at the sequence end, use pad instead.

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

