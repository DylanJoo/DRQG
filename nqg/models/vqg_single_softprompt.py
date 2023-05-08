"""
TODO: Making this module to be the inherited class of vqg_single_dist
"""
import torch
import inspect
from typing import Optional, Tuple, Union, Dict, Any
from transformers import T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from torch import nn
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from utils import kl_weight, kl_loss, sim_loss
import copy
from .prompt import SoftEmbedding

class T5VQG(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config, vae_config=None, tokenizer=None):
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

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

        # Reparameterized initialization
        self.tokenizer = tokenizer
        self.n_soft_prompts = vae_config.n_soft_prompts
        self.d_latent = vae_config.latent_size
        self.vae_kwargs = {
                'annealing_fn': vae_config.annealing_fn, 
                'k': vae_config.k, 'x0': vae_config.x0
        }

        self.prompts = SoftEmbedding(
                wte=self.shared, 
                n_prompts=vae_config.n_soft_prompts,
                hidden_size=config.d_model,
                latent_size=vae_config.latent_size
        )
        self.prompts.set_gaussian_n_samples_for_generation(5)
        self.n_samples = self.prompts.n_samples
        self.samples_mapping = {i: std for i, std in enumerate(self.prompts.std_list)}

        self.encoder.set_input_embeddings(self.prompts)
        self.decoder.set_input_embeddings(self.shared)
        print('Prompt embedding set finished.')

        self.test = nn.Linear(config.d_model, config.d_model)

    def _reparam_inputs(self, input_ids, attn_mask, steps=None):
        """
        Instead of using the original embedding layer,
        replacing it with the SoftEmbedding layer, 
        whcih add the prefix prompt before the input text.
        """
        # input_ids --> input_embeds (with reparameterization)
        inputs_embeds = self.encoder.embed_tokens(
                input_ids.view(-1, input_ids.size()[-1]),
                is_train=(steps is not None),
                **self.vae_kwargs, steps=steps
        )
        # attention_mask --> attention_mask (row expanded)
        if inputs_embeds.size(0) % input_ids.size(0) != 0:
            N = inputs_embeds.size(0) // (attn_mask.size(0)//2)
            attn_mask_ = attn_mask[:attn_mask.size(0)//2, :]
        else:
            N = inputs_embeds.size(0) // attn_mask.size(0)
            attn_mask_ = attn_mask

        # attention_mask --> attention_mask (col expanded)
        soft_attn_mask = torch.ones((attn_mask_.size(0), self.n_soft_prompts))
        attn_mask_ = torch.cat([soft_attn_mask.to(attn_mask_.device), attn_mask_], 1)[:, :512]
        attn_mask_new = attn_mask_.repeat(N, 1)

        return inputs_embeds, attn_mask_new

    def generate(
        self, 
        input_ids=None, 
        attention_mask=None, 
        **kwargs
    ):
        inputs_embeds, attention_mask_new = self._reparam_inputs(
                input_ids, attention_mask, steps=None
        )
        return super().generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask_new, 
                **kwargs
        )

    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        steps: Optional[int] = None,  
        **kwargs
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        # TODO prompt the decoder input 
        if encoder_outputs is None:
            inputs_embeds, attention_mask_new = self._reparam_inputs(
                    input_ids, attention_mask, steps
            )
        else:
            inputs_embeds = None
            attention_mask_new = attention_mask

        # Standard encoder-decoder pipeline
        outputs = super().forward(
                input_ids=None,
                attention_mask=attention_mask_new,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                labels=labels,
                inputs_embeds=inputs_embeds,
                **kwargs
        )

        # Loss
        if labels is not None:
            if steps % 50 == 0:
                print(f"\nNLL: {outputs['loss']}\
                        \nKLD: {self.encoder.embed_tokens.loss_KL}\
                        \nCOSINE: {self.encoder.embed_tokens.loss_COSINE}")
                with torch.no_grad():
                    n = input_ids.size(0)//2
                    input_ids = input_ids[:n, :]
                    attention_mask = attention_mask[:n, :]
                    temp = self.generate(input_ids, attention_mask=attention_mask)
                    labels_reformulate = [l for l in labels[0] if l != -100]
                    print("D2Q+ *", self.tokenizer.decode(labels_reformulate, skip_special_tokens=True))
                    for i in range(self.n_samples):
                        print(f"D2Q ({self.samples_mapping[i]}):", self.tokenizer.decode(temp[i*n], skip_special_tokens=True))
                    labels_reformulate = [l for l in labels[n] if l != -100]
                    print("D2Q- *", self.tokenizer.decode(labels_reformulate, skip_special_tokens=True))

            outputs.loss += self.encoder.embed_tokens.loss_KL

        return outputs

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

