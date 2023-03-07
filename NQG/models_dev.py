import torch
from typing import Optional, Tuple, Union
from transformers import BartPretrainedModel, BartModel, BartConfig
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutput
from torch import nn
from torch.nn import CrossEntropyLoss
from utils import kl_weight, kl_loss
import copy

from transformers.modeling_bart import (
        BART_INPUTS_DOCSTRING, 
        _CHECKPOINT_FOR_DOC,
        _CONFIG_FOR_DOC = "BartConfig",
        _EXPECTED_OUTPUT_SHAPE
)

class BartVAE(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"lm_head.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, config: BartConfig, tokenizer=None, vae_config=None):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

        ## BartForConditionalGeneration
        self.lm_head = nn.Linear(
                config.d_model, 
                self.model.shared.num_embeddings, 
                bias=False
        )
        ## BartForSequenceLabeling
        self.classification_head = BartClassificationHead(
                config.d_model,
                config.d_model,
                config.num_labels,
                config.classifier_dropout
        )

        # VAE 
        self.vae_config = vae_config
        hidden_factor = 1
        self.latent_size = self.vae_config.latent_size
        self.hidden2mean = nn.Linear(
                encoder_config.d_model * hidden_factor, 
                self.vae_config.latent_size
        )
        self.hidden2logv = nn.Linear(
                encoder_config.d_model * hidden_factor, 
                self.vae_config.latent_size
        )
        self.latent2hidden = nn.Linear(
                self.vae_config.latent_size, 
                encoder_config.d_model * hidden_factor
        )
        self.tokenizer = tokenizer

        # Initialize weights and apply final processing
        self.post_init()

        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.out_proj)


    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
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
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        The model for BartVAE models with both generative and discriminative losses.
        Params
        ------

        Returns
        -------
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state
        encoder_hidden_states = outputs.encoder_last_hidden_state

        ## [Generative]
        ## The following codes from `BartForConditionalGeneration`
        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        ## [Discriminative]
        ## The following codes from `BartForSequenceClassification`
        ## But use the bi-encoder architecture
        ## [NOTE] Use the encoder's output or decoder's one
        eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(
                hidden_states.size(0), -1, hidden_states.size(-1)
        )[:, -1, :]

        ## Constrastive learning
        logits = self.classification_head(sentence_representation)

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
