import torch
import inspect
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import BartForConditionalGeneration, BartConfig
from transformers.models.bart.modeling_bart import shift_tokens_right, BartClassificationHead, BartModel
from transformers.modeling_outputs import BaseModelOutput

from torch import nn
from torch.nn import CrossEntropyLoss

from utils import kl_weight, kl_loss
import copy
from .questiongenerator import BartQG
from .modules import EncDecCVAE, InstanceWisePrompt, RelevancePrompt

@dataclass
class Seq2SeqLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    reparam_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    clf_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

def shift_tokens_right_modified(
        input_ids: torch.Tensor, 
        pad_token_id: int, 
        eos_token_id: int
    ):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = pad_token_id 

    # sometimes the last token is eos; thus, replace the last one by eos then.
    shifted_input_ids[input_ids[:, -1].eq(eos_token_id), -1] = eos_token_id
    # since we focus the eos token at the sequence end, use pad instead.

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class BartCVQG(BartQG):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"lm_head.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, config: BartConfig, vqg_config=None):
        super().__init__(config)

        # [Bart encoder-decoder]
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.model = BartModel(config)

        # [Generation]
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # [Classification]
        if vqg_config.add_classification_head:
            self.classification_head = BartClassificationHead(
                config.d_model,
                config.d_model,
                config.num_labels,
                config.classifier_dropout,
            )
        else:
            self.classification_head = None

        # Initialize weights and apply final processing
        self.post_init()

        # soft prompting
        kld_kwargs = {'annealing_fn': vqg_config.annealing_fn}

        if kld_kwargs['annealing_fn'] != 'cyclic':
            kld_kwargs.update({'k': vqg_config.k, 'x0': vqg_config.x0})
        else:
            kld_kwargs.update({'n_total': vqg_config.total_iter, 'n_cycle': vqg_config.n_cycle})

        ## [prompt] 
        self.n_prompts = vqg_config.n_prompts
        self.enc_iwprompts = InstanceWisePrompt(
                wte=self.model.shared,
                hidden_size=config.d_model,
                head_size=vqg_config.head_size,
                length=vqg_config.n_prompts,
                init_idx=vqg_config.used_prompt_idx
        )

        ## [variational]
        self.encdec_cvae = EncDecCVAE(
                wte=self.model.shared,
                hidden_size=config.d_model,
                latent_size=vqg_config.latent_size,
                length=vqg_config.n_labels,
                prefix_length=vqg_config.n_prompts,
                has_compressed_layer=vqg_config.has_compressed_layer,
                pooling=vqg_config.pooling,
                init_idx=vqg_config.used_label_idx,
                learnable_prior=vqg_config.learnable_prior,
                kld_kwargs=kld_kwargs, 
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
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        # Parameters
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        ## [decoder inputs] 
        if labels is not None:
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        ## [Bart encoder]
        if encoder_outputs is None:
            inputs_embeds = torch.cat([
                self.enc_iwprompts(tokens=input_ids),
                self.model.encoder.embed_tokens(input_ids)
            ], 1)
            attention_mask = self.enc_iwprompts.expand_mask(attention_mask) 
            encoder_outputs = self.model.encoder(
                input_ids=input_ids if inputs_embeds is None else None,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # [Prompting with variational encoding]
        hidden_state_prime, reparam_loss = self.encdec_cvae(
                encoder_outputs[0], clf_scores.to(self.device), steps
        )
        encoder_hidden_states = encoder_outputs[0] + hidden_state_prime

        # [Bart decoder] 
        # outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
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

        ## [generation head]
        lm_logits = self.lm_head(decoder_outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        clf_logits = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            if self.classification_head is not None:
                hidden_states = decoder_outputs.last_hidden_state
                decoder_input_ids = shift_tokens_right_modified(
                        labels, 
                        self.config.pad_token_id, 
                        self.config.eos_token_id
                )
                # sanity check for eos tokens
                eos_mask = decoder_input_ids.eq(self.config.eos_token_id).to(hidden_states.device)
                sentence_representation = hidden_states[eos_mask, :]
                clf_logits = self.classification_head(sentence_representation)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            reparam_loss=reparam_loss,
            clf_logits=clf_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
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

        # 3+. 
        model_kwargs['clf_labels'] = encoder_kwargs.pop('clf_labels', None)
        model_kwargs['clf_scores'] = encoder_kwargs.pop('clf_scores', None)
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

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

