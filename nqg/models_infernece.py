import torch
from typing import Optional, Tuple, Union
from transformers import T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from torch import nn
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from utils import kl_weight, kl_loss, hellinger_loss
import copy

class T5VAEForConditionalGeneration(T5ForConditionalGeneration):

    def inference(self, 
        output_positive: Optional[bool] = True, 
        z_input_embeds: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None, 
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        steps: int = None, 
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        # Encode if needed (training, first prediction pass)
        # [NOTE] The encoder output may sometimes could be the same. 
        # Run for tone time
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # [NOTE] substitute the initialized one.
        hidden_states = encoder_outputs[0]

        # Reparameterize with `z_input_embeds`
        ## (1) random interpolated random tensor 
        batch_size, seq_length, d_model = hidden_states.shape
        if z_input_embeds is not None:
            r = torch.randn([pn_boundary, 1, self.latent_size]).to(hidden_states.device)
        else:
            r = z_input_embeds

        ## (2) positive/negative query generation
        if output_positive:
            pmean = self.hidden2pmean(hidden_states[:, :1, :])
            plogv = self.hidden2plogv(hidden_states[:, :1, :])
            pstd = torch.exp(0.5 * plogv)
            z = r * pstd + pmean
            z_hidden = self.latent2hidden(z)
        else:
            nmean = self.hidden2nmean(hidden_states[:, :1, :])
            nlogv = self.hidden2nlogv(hidden_states[:, :1, :])
            nstd = torch.exp(0.5 * nlogv)
            z = r * nstd + nmean
            z_hidden = self.latent2hidden(z)

        zeros = torch.zeros(batch_size, seq_length-1, d_model).to(hidden_states.device)
        residuals = torch.cat((z_hidden, zeros), 1)
        encoder_outputs['last_hidden_state'] = hidden_states + residuals

        # [NOTE] Some codes in `forward()` are discaraded. 
        # [TODO] check `generate`
        return super().gererate(
                encoder_outputs=encoder_output,
        )
        # Generate 
        # Decode
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     inputs_embeds=decoder_inputs_embeds,
        #     past_key_values=past_key_values,
        #     encoder_hidden_states=hidden_states,
        #     encoder_attention_mask=attention_mask,
        #     head_mask=decoder_head_mask,
        #     cross_attn_head_mask=cross_attn_head_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        # sequence_output = decoder_outputs[0]

        # self.lm_head = self.lm_head.to(self.encoder.first_device)
        # sequence_output = sequence_output.to(self.lm_head.weight.device)
