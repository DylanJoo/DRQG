import torch
from typing import List, Optional, Tuple, Union
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
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.prompts = SoftEmbedding(
                wte=self.model.shared, 
                n_prompts=vae_config.n_soft_prompts,
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
        self.post_init()

        # input_ids: torch.LongTensor = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # decoder_input_ids: Optional[torch.LongTensor] = None,
        # decoder_attention_mask: Optional[torch.LongTensor] = None,
        # head_mask: Optional[torch.Tensor] = None,
        # decoder_head_mask: Optional[torch.Tensor] = None,
        # cross_attn_head_mask: Optional[torch.Tensor] = None,
        # encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        # decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # labels: Optional[torch.LongTensor] = None,
        # use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,

    def _reparam_inputs(self, input_ids, attn_mask, steps=None):
        # input_ids --> input_embeds (with reparameterization)
        inputs_embeds = self.model.encoder.embed_tokens(
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
        **kwargs
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        if encoder_outputs is None:
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
                attetnion_mask=attention_mask_new,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                labels=labels,
                inputs_embeds=inputs_embeds,
                **kwargs
        )

        # Loss
        if labels is not None:
            if steps % 50 == 0:
                # self.eval()
                print(f"\nNLL: {outputs['loss']}\
                        \nKLD: {self.model.encoder.embed_tokens.loss_KL}")
                with torch.no_grad():
                    n = input_ids.size(0)//2
                    input_ids = input_ids[:n, :]
                    attention_mask = attention_mask[:n, :]
                    out = self.generate(
                            input_ids, attention_mask=attention_mask, 
                            return_dict_in_generate=True,
                            output_scores=True
                    )
                    temp = out.sequences
                    logits = out.scores
                    labels_reformulate = [l for l in labels[0] if l != -100]
                    print("D2Q+ *", self.tokenizer.decode(labels_reformulate, skip_special_tokens=True))
                    for i in range(self.n_samples):
                        ## texts
                        print(f"D2Q ({self.samples_mapping[i]:<3}):", 
                                self.tokenizer.decode(temp[i*n], skip_special_tokens=True)
                        )
                        ## probs (debug)
                        p = []
                        for j in range(len(logits)):
                            p.append(round(torch.nn.functional.softmax(logits[j][i]).max().item(), 2))
                        print("------->:", p)

                    labels_reformulate = [l for l in labels[n] if l != -100]
                    print("D2Q- *", self.tokenizer.decode(labels_reformulate, skip_special_tokens=True))
                self.train()

            # add reparameterize
            outputs.loss += self.model.encoder.embed_tokens.loss_KL

        # TODO: Bart clf
        # hidden_states = outputs[0]  # last hidden state
        #
        # eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)
        #
        # if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
        #     raise ValueError("All examples must have the same number of <eos> tokens.")
        # sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
        #     :, -1, :
        # ]
        return outputs

