import torch
from typing import Optional, Tuple, Union
from transformers import T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from torch import nn
from torch.nn import CrossEntropyLoss
from utils import kl_weight, kl_loss
import copy

class T5VAEForConditionalGeneration(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config, tokenizer=None, vae_config=None):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        # ========== VAE setting ==========
        self.vae_config = vae_config
        self.latent_size = self.vae_config.latent_size
        self.hidden2pmean = nn.Linear(encoder_config.d_model, self.vae_config.latent_size)
        self.hidden2nmean = nn.Linear(encoder_config.d_model, self.vae_config.latent_size)
        self.hidden2plogv = nn.Linear(encoder_config.d_model, self.vae_config.latent_size)
        self.hidden2nlogv = nn.Linear(encoder_config.d_model, self.vae_config.latent_size)
        self.latent2hidden = nn.Linear(self.vae_config.latent_size, encoder_config.d_model)
        self.tokenizer = tokenizer
        # ========== VAE setting ==========

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

    def forward(
        self,
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

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        """
        Sinece the SVAE is adopted on 'sentence embedding' VAE, 
        this model use L tokens embedding reconstruction VAE process instead.
        hidden_states (B L H)
        """
        # ======T5 VAE ===== 
        # REPARAMETERIZATION
        batch_size, seq_length, d_model = hidden_states.shape
        pn_boundary = batch_size // 2

        # [NOTE] Transform it into a single vector (i.e., d dimensions x 1 tokens) with mask
        aggregation_mask = torch.tensor(
            [[1] + [0] * (seq_length-1)]
        ).transpose(0, 1).repeat(1, d_model).repeat(batch_size, 1, 1).to(hidden_states.device)
        hidden_states = hidden_states * aggregation_mask
        encoder_outputs.hidden_states = hidden_states

        # [NOTE] Regarding each element (i.e., d dimensions x |L| tokens)
        # None

        mean = self.hidden2pmean(hidden_states[:, :1, :])
        logv = self.hidden2plogv(hidden_states[:, :1, :])
        std = torch.exp(0.5 * logv)
        z = torch.randn([batch_size//2, 1, self.latent_size]).to(hidden_states.device)
        z = z * std + mean
        hidden_states[:pn_boundary, :1, :] += self.latent2hidden(z)

        mean = self.hidden2nmean(hidden_states[:, :1, :])
        logv = self.hidden2nlogv(hidden_states[:, :1, :])
        std = torch.exp(0.5 * logv)
        z = torch.randn([batch_size//2, 1, self.latent_size]).to(hidden_states.device)
        z = z * std + mean
        hidden_states[:pn_boundary, :1, :] += self.latent2hidden(z)

        # ======T5 VAE =====

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
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
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)


        lm_logits = self.lm_head(sequence_output)

        # ======T5 VAE ===== 
        # Calculate losses
        loss = None
        loss_ce = 0
        loss_kl = 0
        loss_kl_w = 0

        if labels is not None:
            # nll loss
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss_ce = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            # kl loss (vae loss)
            loss_kl_w = kl_weight(
                    self.vae_config.annealing_fn, steps, self.vae_config.k, self.vae_config.x0
            )
            loss_kl = kl_loss(
                    logv.view(-1, self.latent_size), mean.view(-1, self.latent_size)
            )

            loss = loss_ce + loss_kl * loss_kl_w

            if steps % 20 == 0:
                print(f"\nNLL: {loss_ce}\
                        \nKL: {loss_kl * loss_kl_w} = {loss_kl} * {loss_kl_w}")
                with torch.no_grad():
                    temp=self.generate(input_ids)
                    print("1:", self.tokenizer.decode(temp[0], skip_special_tokens=True))
                    temp=self.generate(input_ids, encoder_outputs=encoder_outputs)
                    print("2:", self.tokenizer.decode(temp[0], skip_special_tokens=True))
                    labels_reformulate = [l for l in labels[0] if l != -100]
                    print("*", self.tokenizer.decode(labels_reformulate, skip_special_tokens=True))
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

