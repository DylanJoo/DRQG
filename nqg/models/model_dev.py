import torch
import inspect
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import BartForConditionalGeneration, BartConfig, BartModel
from transformers.models.bart.modeling_bart import BartClassificationHead
from transformers.modeling_outputs import Seq2SeqLMOutput

from torch import nn
from torch.nn import CrossEntropyLoss

from utils import kl_weight, kl_loss
import copy
from .prompt import SoftEmbedding

class BartVQG(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"lm_head.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, config: BartConfig, vae_config=None, tokenizer=None):
        super().__init__(config)

        # Reparameterized initialization
        self.tokenizer = tokenizer
        self.n_soft_prompts = vae_config.n_soft_prompts
        self.vae_kwargs = {
                'annealing_fn': vae_config.annealing_fn, 
                'k': vae_config.k, 'x0': vae_config.x0
        }

        # soft prompting
        self.prompts = SoftEmbedding(
                wte=self.model.shared, 
                n_prompts=vae_config.n_soft_prompts,
                initialize_from_vocab=vae_config.initialize_from_vocab,
                hidden_size=config.d_model,
                latent_size=vae_config.latent_size
        )
        self.prompts.set_gaussian_n_samples_for_generation(5)
        self.n_samples = self.prompts.n_samples
        self.samples_mapping = {i: std for i, std in enumerate(self.prompts.std_list)}

        self.model.encoder.set_input_embeddings(self.prompts)
        self.model.decoder.set_input_embeddings(self.model.shared)
        print('Prompt embedding set finished.')

        # sequence classification task
        # self.classification_head = BartClassificationHead(
        #     config.d_model,
        #     config.d_model,
        #     config.num_labels,
        #     config.classifier_dropout,
        # )

        # Initialize weights and apply final processing

    def _reparam_inputs(self, input_ids, attn_mask, steps=None):
        # input_ids --> input_embeds (with reparameterization)
        inputs_embeds = self.model.encoder.embed_tokens(
                input_ids.view(-1, input_ids.size()[-1]),
                is_train=(steps is not None),
                **self.vae_kwargs, steps=steps
        ) * self.model.encoder.embed_scale
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
        input_ids_eval: torch.LongTensor = None,
        attention_mask_eval: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        if encoder_outputs is None: # the first momnet when generating 
            inputs_embeds, attention_mask_new = self._reparam_inputs(
                    input_ids, attention_mask, steps
            )
        else:
            inputs_embeds = None
            attention_mask_new = attention_mask

        # standard enc-dec pipeline
        # TODO add classfication task head here
        outputs = super().forward(
                input_ids=None, 
                attention_mask=attention_mask_new,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_head_mask,
                head_mask=head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
        )

        # Loss
        if labels is not None:
            if steps % 50 == 0:
                # self.eval()
                print(f"\nNLL: {outputs['loss']}\
                        \nKLD: {self.model.encoder.embed_tokens.loss_KL}")
                with torch.no_grad():
                    # generate the normal one
                    n=input_ids_eval.size()[0]
                    out = self.generate_(
                            input_ids_eval, 
                            attention_mask=attention_mask_eval, 
                            return_dict_in_generate=True,
                            output_scores=True,
                            num_beams=5
                    )
                    temp = out.sequences
                    logits = out.scores
                    labels_reformulate = [l for l in labels[0] if l != -100]
                    print("D2Q+ *", self.tokenizer.decode(labels_reformulate, skip_special_tokens=True))
                    for i in range(self.n_samples):
                        print(f"D2Q ({self.samples_mapping[i]:<3}):", 
                                self.tokenizer.decode(temp[i*n], skip_special_tokens=True)
                        )
                        # p = []
                        # for j in range(len(logits)):
                        #     p.append(round(torch.nn.functional.softmax(logits[j][i]).max().item(), 2))
                        # print("------->:", p)

                    labels_reformulate = [l for l in labels[n] if l != -100]
                    print("D2Q- *", self.tokenizer.decode(labels_reformulate, skip_special_tokens=True))
                # self.train()

            # add reparameterize
            outputs.loss += self.model.encoder.embed_tokens.loss_KL

        return outputs

    # TODO: Bart clf
    # def _run_classification(self, hidden_states):
    #     eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)
    #
    #     if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
    #         raise ValueError("All examples must have the same number of <eos> tokens.")
    #     sentence_representation = hidden_states[eos_mask, :].view(
    #             hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]
    #
    #     logits = self.classification_head(sentence_representation)
    #
    #     # setting 1: similarity (ideally, first half contradicts the other)
    #     loss_fct = CrossEntropyLoss()
    #     loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
    #     return loss

    def generate_(
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

