"""
TODO: Making this module to be the inherited class of vqg_single_dist
"""
import torch
from typing import Optional, Tuple, Union
from transformers import T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from torch import nn
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from utils import kl_weight, kl_loss
import copy

class T5VQG(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config, vae_config, tokenizer=None, debug=None):
        super().__init__(config)
        # Debugging 
        self.debug = debug # model_args.mode

        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self._initialize_variational_modules(config, vae_config)

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

        self.tokenizer = tokenizer

    def set_gaussian_n_samples_for_generation(self, n_side: int):
        self.std_list = list(range(-n_side, n_side+1, 1))
        self.n_samples = 1 + 2*n_side

    def _initialize_variational_modules(self, t5_config, config):
        """ [TODO] Inheritance"""
        self.latent_size = config.latent_size
        latent_size = config.latent_size
        self.hidden2mean = nn.Linear(t5_config.d_model, latent_size, bias=False)
        self.hidden2logv = nn.Linear(t5_config.d_model, latent_size, bias=False)
        self.latent2hidden = nn.Linear(latent_size, t5_config.d_model, bias=False)
        self.vae_config = config

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

            # forward with variational module
            hidden_states, loss_variational = self.forward_variational(
                    encoder_outputs, steps=steps
            )
            loss_reparam = loss_variational

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

            # forward with variational module
            hidden_states, loss_variational = self.forward_variational(
                    encoder_outputs, steps=steps
            )
            loss_reparam = loss_variational
        else: # in inference mode
            loss_reparam = 0
            hidden_states = encoder_outputs[0]

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

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels_pos = copy.deepcopy(labels)
            labels_neg = copy.deepcopy(labels)

            pn_boundary = labels.size(0) // 2
            labels_pos[pn_boundary:, :] = -100 # mask NQG (the latter half)
            labels_neg[:pn_boundary, :] = -100 # mask PQG (the former half)
            loss_ce_pos = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels_pos.view(-1))
            loss_ce_neg = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels_neg.view(-1))

            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            loss_ce = (loss_ce_pos + loss_ce_neg) / 2

            # add varaiational losses
            if steps % 50 == 0:
                print(f"\nNLL: (positive) {loss_ce_pos}\t(negative) {loss_ce_neg}\
                        \nKLD: {loss_reparam[0]} (weight) {loss_reparam[1]}")
                # inferece during training
                if steps % 50 == 0:
                    with torch.no_grad():
                        temp=self.generate(input_ids)

                        labels_reformulate = [l for l in labels[0] if l != -100]
                        print("D2Q+:", self.tokenizer.decode(temp[0], skip_special_tokens=True))
                        print("D2Q+*", self.tokenizer.decode(labels_reformulate, skip_special_tokens=True))

                        n=input_ids.shape[0]//2
                        labels_reformulate = [l for l in labels[n] if l != -100]
                        print("D2Q-:", self.tokenizer.decode(temp[n], skip_special_tokens=True))
                        print("D2Q-*", self.tokenizer.decode(labels_reformulate, skip_special_tokens=True))

            loss = loss_ce + loss_reparam[0]

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

    def forward_variational(
        self, 
        encoder_outputs,
        steps: Optional[int] = None
    ) -> Optional[torch.FloatTensor]:
        """ The standard version of forward passing (training) 
        :param encoder_outputs: the T5 encoder's (T5Stack) output.
        :return hidden_states: the revised hidden_states
        """
        hidden_states = encoder_outputs[0]
        batch_size, seq_length, d_model = hidden_states.shape
        pn_boundary = batch_size // 2

        # Get controller tokens
        embeds = hidden_states[:, :1, :]
        
        # Polarity
        mean = self.hidden2mean(embeds[pn_boundary:, :, :])
        logv = self.hidden2logv(embeds[pn_boundary:, :, :])
        std = torch.exp(0.5 * logv)
        r = torch.randn([pn_boundary, 1, self.latent_size], device=hidden_states.device)
        z = r * std + mean

        if self.debug == 1:
            # append prompty token
            # h_prime (single) with h
            positive = self.latent2hidden(mean) # without noise
            negative = self.latent2hidden(z)    # with noise
            residuals = torch.cat((positive, negative), 0)
            hidden_states = hidden_states[:, 1:, :]
            hidden_states = torch.cat((residuals, hidden_states), 1)
        else:
            # residual learning
            # [h_prime (single) + h (single)] with h
            positive = self.latent2hidden(mean) # without noise
            negative = self.latent2hidden(z)    # with noise
            zeros = torch.zeros(batch_size, seq_length-1, d_model).to(hidden_states.device)
            residuals = torch.cat((torch.cat((positive, negative), 0), zeros), 1)
            hidden_states = hidden_states + residuals

        # Compute variational losses
        loss_reparam = self.compute_loss_reparam(mean, logv, steps)

        return hidden_states, (loss_reparam)

    def compute_loss_reparam(self, mean, logv, steps):
        loss_kl_w = 1
        loss_kl = kl_loss(logv.view(-1, self.latent_size), mean.view(-1, self.latent_size))

        if steps is not None:
            loss_kl_w = kl_weight(
                self.vae_config.annealing_fn, 
                steps, 
                self.vae_config.k, 
                self.vae_config.x0
            )
        if self.debug == 2:
            loss_kl_w = 1

        return loss_kl_w * loss_kl, loss_kl_w

