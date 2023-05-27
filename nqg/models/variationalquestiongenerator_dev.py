import torch
import inspect
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import BartForConditionalGeneration, BartConfig
from transformers.models.bart.modeling_bart import shift_tokens_right, BartClassificationHead, BartEncoder, BartDecoder, BartModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import ModelOutput

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
        SoftAdaptiveEmbedding2, 
)

PROMPT_EMBEDS = {
        'basic': SoftEmbedding,
        'static': SoftStaticEmbedding,
        'adaptive': SoftAdaptiveEmbedding,
        'attentive': SoftEmbeddingWithPooler,
        'adaptive2': SoftAdaptiveEmbedding2,
        # 'residual': SoftResidualEmbedding
}

@dataclass
class Seq2SeqLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
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
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.model = BartModel(config)

        # [Generation]
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # [Classification]
        if vqg_config.add_classification_head:
            self.classification_head = BartClassificationHead(
                config.d_model,
                config.d_model,
                1,
                config.classifier_dropout,
            )
        else:
            self.classification_head = None

        # Initialize weights and apply final processing
        self.post_init()

        # Reparameterized initialization
        self.n_soft_prompts = vqg_config.n_soft_prompts

        # soft prompting
        ## config
        kwargs = {
                'annealing_fn': vqg_config.annealing_fn, 
                'k': vqg_config.k, 'x0': vqg_config.x0,
                'has_compressed_layer': vqg_config.has_compressed_layer
        }

        ## prompt layer
        self.enc_prompts = PROMPT_EMBEDS[vqg_config.pooling](
                wte=self.model.shared, 
                hidden_size=config.d_model,
                latent_size=vqg_config.latent_size,
                initialize_from_vocab=vqg_config.initialize_from_vocab,
                used_prompt_idx=vqg_config.used_prompt_idx,
                n_prompts=vqg_config.n_soft_prompts,
                pooler=None,
                adaptive_pooling=vqg_config.adaptive_pooling,
                **kwargs
        )
        self.model.encoder.set_input_embeddings(self.enc_prompts)
        self.model.decoder.set_input_embeddings(self.model.shared)

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

        # Bart Model
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
            # [NOTE] change clf labels to clf scores
            inputs_embeds, attention_mask_new = self.reparam_inputs(
                    input_ids, attention_mask, steps, clf_scores
            )
            encoder_outputs = self.model.encoder(
                input_ids=input_ids if inputs_embeds is None else None,
                attention_mask=attention_mask_new,
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
            inputs_embeds = None
            attention_mask_new = attention_mask
        else:
            inputs_embeds = None
            attention_mask_new = attention_mask

        # [Bart decoder] 
        # outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask_new,
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
            clf_logits=clf_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
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

