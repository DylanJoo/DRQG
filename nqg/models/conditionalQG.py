from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import inspect
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BartConfig 
from transformers.models.bart.modeling_bart import (
        BartModel,
        shift_tokens_right, 
        BartClassificationHead
)
from transformers.modeling_outputs import BaseModelOutput

from .outputs import Seq2SeqCVQGOutput
from .questiongenerator import BartQG
from .controller import DocRelPrompt

import copy

class DocRelBartQG(BartQG):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"lm_head.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]
    def __init__(self, config: BartConfig, cvqg_config=None, **kwargs):
        super().__init__(config)
        self.model = BartModel(config)

        # [generation] 
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)   

        # [classification]
        if cvqg_config.add_classification_head:
            self.classification_head = BartClassificationHead(
                config.d_model, config.d_model, 2,
                config.classifier_dropout,
            )
        else:
            self.classification_head = None

        self.post_init()

        # [prompt] 
        self.controller = DocRelPrompt(
                wte=self.model.shared, 
                hidden_size=config.d_model,
                init_idx=cvqg_config.prompts_idx,
                lbl_init_idx=cvqg_config.label_prompts_idx,
        )
        self.batch_size = kwargs.pop('batch_size', None)

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
            ## [Encoder prompt] 
            inputs_embeds = self.controller(clf_scores, input_ids=input_ids)
            attention_mask = self.controller.expand_mask(attention_mask)

            ## [Bart encoder]
            encoder_outputs = self.model.encoder(
                input_ids=input_ids if inputs_embeds is None else None,
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

        ## [EncoderDecoder prompt wrapper]
        encoder_hidden_states = encoder_outputs[0]

        ## setting 0 # very failed
        encoder_hidden_state = encoder_hidden_states

        ## setting 1 & 2: 
        ### remove prompts or not 
        # encoder_hidden_states = encoder_hidden_states[:, self.controller.length:, :]

        # mean pooling
        encoder_hidden_state = encoder_hidden_states.mean(1)[:, None, :]
        attention_mask=None

        # cls pooling
        # encoder_hidden_states = encoder_hidden_states[:, self.controller.length:, :]
        # encoder_hidden_state = encoder_hidden_states[:, :1, :]

        # standard enc-dec pipeline
        decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_state,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )
        lm_logits = self.lm_head(decoder_outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = 0
        docibn_loss = 0
        reparam_loss = 0
        clf_logits = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                    lm_logits.view(-1, self.config.vocab_size), 
                    labels.view(-1)
            )

            # doc ibn
            docibn_loss = self.controller.calculate_src_ibn_loss(
                    encoder_hidden_state, self.batch_size, norm=True
            )

            # [discriminator]
            if self.classification_head is not None:
                hidden_states = decoder_outputs.last_hidden_state
                decoder_input_ids_ = shift_tokens_right_modified(
                        labels, 
                        self.config.pad_token_id, 
                        self.config.eos_token_id
                )
                eos_mask = decoder_input_ids_.eq(self.config.eos_token_id).to(hidden_states.device)
                sentence_representation = hidden_states[eos_mask, :]
                clf_logits = self.classification_head(sentence_representation) # B 2

        return Seq2SeqCVQGOutput(
                loss=masked_lm_loss, 
                reparam_loss=reparam_loss,
                docibn_loss=docibn_loss,
                logits=lm_logits, 
                clf_logits=clf_logits,
                past_key_values=decoder_outputs.past_key_values, 
                decoder_hidden_states=decoder_outputs.hidden_states, 
                decoder_attentions=decoder_outputs.attentions, 
                cross_attentions=decoder_outputs.cross_attentions, 
        )

    # tune for encoder-decoder when generation
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
        encoder_signature.add('clf_labels')
        encoder_signature.add('clf_scores')
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor

        # 3+. retrieva labels and scores
        clf_labels = encoder_kwargs.pop('clf_labels', None)
        clf_scores = encoder_kwargs.pop('clf_scores', None)

        ## add the additional layer
        encoder_kwargs['inputs_embeds'] = self.controller(clf_scores, input_ids=encoder_kwargs['input_ids'])
        encoder_kwargs['input_ids'] = None
        encoder_kwargs['attention_mask'] = self.controller.expand_mask(encoder_kwargs['attention_mask'])
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)
        model_kwargs["attention_mask"] = encoder_kwargs['attention_mask']

        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "clf_labels": kwargs.pop('clf_labels', None),
            "clf_scores": kwargs.pop('clf_scores', None),
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

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

