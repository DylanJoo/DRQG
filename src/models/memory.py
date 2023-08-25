import torch
from torch import nn
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import T5Config 
from transformers.models.bart.modeling_bart import (
    BartModel,
    shift_tokens_right, 
    BartClassificationHead
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from _base import FlanT5

class MemoryFlanT5(FlanT5):

    def __init__(self, memory_type='enlarge', **kwargs):
        """ add a memory projection layer.

        :param: memory_type ['enlarge', 'layer_wise'] 
         enlarge is up-projection from `d_model` to `d_model*n_layers`.
         layer_wise is projection from `d_mdoel` to `d_model` layer-wise.
        """
        super().__init__(self, **kwargs)
        if memory_type == 'enlarge':
            d_memory = config.num_decoder_layers 
            d_memory *= config.d_model
        elif memory_proj == 'layer_wise':
            d_memory *= config.d_model

        self.embed_size_per_head = config.d_model // config.num_heads
        self.memory_type = memory_type
        self.memory_layer = nn.Linear(condig.d_model, d_memory, bias=False)

    def memory_projection(self, encoder_outputs):
        """ this projection is to extract the feature from encoder outputs; 
        it performs in two ways, last hidden state or all hidden states.
        """
        B = encoder_outputs[0].shape[0]
        Nh = self.config.decoder_attention_heads

        if self.memory_type == 'enlarge':
            memory = self.memory_layer(encoder_outputs['last_hidden_states'])
        elif self.memory_type == 'layer_wise':
            memory = self.memory_layer(torch.cat( layer[:-1, :, :] \
                    for layer in encoder_outputs['hidden_states']
            ))
        memory = memory.view(
                B, M,
                self.config.decoder_layers,
                self.config.num_heads,
                self.embed_size_per_head
        ).permute(2, 0, 3, 1, 4)

        return tuple((None, None, ca, ca) for ca in memory)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        **kwargs
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        """
        :param input_ids: input tokenized ids
        :param attention_mask: tokenized attention mask
        :param encoder_outputs: the encoder's output class during training or 
            first prediction pass.
        :param past_key_values: the cached key value states during decoding
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # [NOTE] Memory projection
        ## project right after the encoding was done.
        if encoder_outputs is not None and past_key_values is not None:
            past_key_values = self.memory_projection(encoder_outputs)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
