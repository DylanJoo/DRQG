"""
TODO: Making this module to be the inherited class of vqg_single_dist
"""
import torch
import inspect
from typing import Optional, Tuple, Union, Dict, Any
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from torch import nn
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from utils import kl_weight, kl_loss
import copy

class SoftEmbedding(nn.Module):

    def get_KL_loss(self):
        return self.kld_loss * self.kld_weight

    def set_gaussian_range(self, std_list):
        self.std_list = std_list

    def __init__(
            self,
            wte: nn.Embedding,
            hidden_size: int = 768, 
            latent_size: int = 128, 
            n_prompts: int = 1,
            initialize_from_vocab: bool = False, 
            pooler: Optional[object] = None,
            **kwargs
        ):
        super(SoftEmbedding, self).__init__()
        self.n_prompts = n_prompts
        self.orig_embeds = wte

        if initialize_from_vocab:
            self.prompt_embeds = nn.Parameter(
                    self.orig_embeds.weight[-self.n_prompts:].clone().detach()
            )
        else:
            self.prompt_embeds = nn.Parameter(
                    torch.rand((self.n_prompts, hidden_size), device=wte.weight.device)-0.5
            )

        self.hidden2mean = nn.Linear(hidden_size, latent_size, bias=False)
        self.hidden2logv = nn.Linear(hidden_size, latent_size, bias=False)
        self.latent2hidden = nn.Linear(latent_size, hidden_size, bias=False)
        self.latent_size = latent_size
        self.pooler = pooler
        self.kld_kwargs = kwargs

    def forward(self, tokens, is_train=False, steps=1):
        """ `training` and `evaluation/prediction` phrases have diff setups. 
        The training batch contains both (specified) batch_size * 2 
        as exact mini-batch during training; while evaluation batch is unmodified."""

        batch_size, seq_length = tokens.shape

        # bos = self.orig_embeds(torch.tensor([1], device=tokens.device)).unsqueeze(0)
        # e_source = self.orig_embeds(tokens[:, 1:])        

        e_source = self.orig_embeds(tokens)
        e_prompt = self.prompt_embeds.unsqueeze(0) 

        if is_train:
            # (1, Np, H)-->(1, Np, Hz)-->(1*2, Np, H)
            mean = self.hidden2mean(e_prompt)
            logv = self.hidden2logv(e_prompt)
            std = torch.exp(0.5*logv)
            r = torch.randn(mean.shape, device=e_source.device)
            z = torch.cat([mean, mean+r*std], 0)
            e_prompt_prime = self.latent2hidden(z) 

            e_input = torch.cat([
                torch.repeat_interleave(e_prompt_prime, batch_size//2, dim=0),
                e_source 
            ], 1)

            # compute loss
            loss = kl_loss(logv.view(-1, self.latent_size),
                           mean.view(-1, self.latent_size))
            self.kld_loss = loss
            self.kld_weight = kl_weight(**self.kld_kwargs, steps=steps)

        else: 
            mean = self.hidden2mean(e_prompt)
            logv = self.hidden2logv(e_prompt)
            std = torch.exp(0.5*logv)
            z = torch.cat([mean+std*i for i in self.std_list], 0)
            e_prompt_prime = self.latent2hidden(z)

            e_input = torch.cat([
                    torch.repeat_interleave(e_prompt_prime, batch_size, dim=0),
                    e_source.repeat(len(self.std_list), 1, 1)
            ], 1)

            self.kld_loss = 0
            self.kld_weight = 0

        # return torch.cat([bos.repeat(e_input.size()[0], 1, 1), e_input], 1)
        return e_input

class SoftAdaptiveEmbedding(SoftEmbedding):

    def forward(self, tokens, is_train=False, steps=1):
        batch_size, seq_length = tokens.shape

        # bos = self.orig_embeds(torch.tensor([1], device=tokens.device)).unsqueeze(0)
        # e_source = self.orig_embeds(tokens[:, 1:])        
        e_source = self.orig_embeds(tokens)
        e_prompt = self.prompt_embeds.unsqueeze(0) 

        # mean pooling
        # e_pooled = torch.mean(e_source, dim=1).unsqueeze(1)

        # max pooling
        e_pooled = torch.max(e_source, dim=1).values.unsqueeze(1)

        e_adaprompt = e_prompt + e_pooled
        # B, n_prompts, hidden

        # Reparameterize
        if is_train: # variational with gaussian noises
            mean = self.hidden2mean(e_adaprompt[:(batch_size//2), :, :])
            logv = self.hidden2logv(e_adaprompt[:(batch_size//2), :, :])
            std = torch.exp(0.5*logv)
            r = torch.randn(mean.shape, device=e_source.device)
            z = torch.cat([mean, mean+r*std], 0) 
            e_adaprompt_prime = self.latent2hidden(z) 

            # Concat z to original embeddings
            e_input = torch.cat([e_adaprompt_prime, e_source], 1)

            # compute loss
            loss = kl_loss(
                    logv.view(-1, self.latent_size),
                    mean.view(-1, self.latent_size)
            )
            self.kld_loss = loss
            self.kld_weight = kl_weight(**self.kld_kwargs, steps=steps)

        else: 
            mean = self.hidden2mean(e_adaprompt)
            logv = self.hidden2logv(e_adaprompt)
            std = torch.exp(0.5*logv)
            z = torch.cat([mean+std*i for i in self.std_list], 0)
            e_adaprompt_prime = self.latent2hidden(z)

            # Concat z to original embeddings
            e_input = torch.cat([
                e_adaprompt_prime,
                e_source.repeat(len(self.std_list), 1, 1)
            ], 1)
            self.kld_loss = 0
            self.kld_weight = 0

        # return torch.cat([bos.repeat(e_input.size()[0], 1, 1), e_input], 1)
        return e_input

class SoftAttentiveEmbedding(SoftEmbedding):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.pooler is not None, \
                'the pooler was not succesfully assigned.'
        print(self.pooler)

    def forward(self, tokens, is_train=False, steps=1):
        e_input = super().forward(tokens, is_train, steps)
        pooled_output = self.pooler(e_input, None, None)[0]
        return pooled_output
