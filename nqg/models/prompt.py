"""
TODO: Making this module to be the inherited class of vqg_single_dist
"""
import torch
import inspect
from typing import Optional, Tuple, Union, Dict, Any
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from torch import nn
from utils import kl_weight, kl_loss, PairwiseCELoss
import copy

class SoftEmbedding(nn.Module):

    def get_pairwise_loss(self):
        return self.pairwise_loss

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
        # [NOTE] If prepending <s> is needed
        self.n_prompts = n_prompts
        self.orig_embeds = wte

        if initialize_from_vocab:
            self.prompt_embeds = nn.Parameter(
                    self.orig_embeds.weight[-self.n_prompts:].clone().detach()
            )
        else:
            self.prompt_embeds = nn.Parameter(
                    torch.randn((self.n_prompts, hidden_size), device=wte.weight.device)
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

        bos = self.orig_embeds(torch.tensor([1], device=tokens.device)).unsqueeze(0)
        e_source = self.orig_embeds(tokens)
        e_prompt = self.prompt_embeds.unsqueeze(0) 

        if is_train:
            # (1, Np, H)-->(1, Np, Hz)-->(1*2, Np, H)
            mean = self.hidden2mean(e_prompt)
            logv = self.hidden2logv(e_prompt)
            std = torch.exp(0.5*logv)
            r = torch.randn(mean.shape, device=e_source.device)
            z = torch.cat([mean, mean+r*std], 0)
            e = self.latent2hidden(z) 

            # [reshape]
            e = e.repeat_interleave(batch_size//2, dim=0)
            # e_source was done
            bos = bos.repeat(batch_size, 1, 1)

            # e_input = torch.cat([bos, e, e_source[:, 1:, :]], 1)
            e_input = torch.cat([e_source, e], 1)

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
            e = self.latent2hidden(z)

            # [reshape]
            e = e.repeat_interleave(batch_size, dim=0)
            e_source = e_source.repeat(len(self.std_list), 1, 1)
            bos = bos.repeat(batch_size*len(self.std_list), 1, 1)

            # e_input = torch.cat([bos, e, e_source[:, 1:, :]], 1)
            e_input = torch.cat([e_source, e], 1)

            self.kld_loss = 0
            self.kld_weight = 0

        return e_input

class SoftAdaptiveEmbedding(SoftEmbedding):
    """
    [ORIG] Adopt the pooling layer before reparameterization.
    [NOTE] Adopt the pooling layer after reparameterization.
        - 1. max pooling with mulitplication
        - 2. max pooling with addition
        - 3. self attention layer with multiplcation
    """
    def forward(self, tokens, is_train=False, steps=1):
        batch_size, seq_length = tokens.shape

        # e_source = self.orig_embeds(tokens[:, 1:])        
        e_source = self.orig_embeds(tokens)
        e_prompt = self.prompt_embeds.unsqueeze(0) 

        # [pooling]
        ## mean
        e_pooled = torch.mean(e_source, dim=1).unsqueeze(1)
        ## max (2B, 1, H)
        # e_pooled = torch.max(e_source, dim=1).values.unsqueeze(1)

        # [reparameterize]
        if is_train: # variational with gaussian noises
            mean = self.hidden2mean(e_pooled[:(batch_size//2)])
            logv = self.hidden2logv(e_pooled[:(batch_size//2)])
            std = torch.exp(0.5*logv)
            r = torch.randn(mean.shape, device=e_source.device)
            z = torch.cat([mean, mean+r*std], 0) 
            e = self.latent2hidden(z) 

            # [reshape]
            e = e + e_prompt.repeat(batch_size, 1, 1)

            # Concat z to original embeddings
            e_input = torch.cat([e, e_source], 1)

            # compute loss
            loss = kl_loss(
                    logv.view(-1, self.latent_size),
                    mean.view(-1, self.latent_size)
            )
            self.kld_loss = loss
            self.kld_weight = kl_weight(**self.kld_kwargs, steps=steps)

            # [debug] in-batch loss
            # 2B L H # 2B N H --> 2B H N
            # 2B L H x 2B H N = 2B L N 
            # scores = (e_source @ e.permute(0, 2, 1)).max(2).values.sum(1)
            # self.pairwise_loss = PairwiseCELoss(scores)

        else: 
            mean = self.hidden2mean(e_pooled)
            logv = self.hidden2logv(e_pooled)
            std = torch.exp(0.5*logv)
            z = torch.cat([mean+std*i for i in self.std_list], 0)
            e = self.latent2hidden(z) 

            # [reshape]
            e = e + e_prompt.repeat(batch_size*len(self.std_list), 1, 1)
            e_source = e_source.repeat(len(self.std_list), 1, 1)

            # Concat z to original embeddings
            e_input = torch.cat([e, e_source], 1)

            # compute loss
            self.kld_loss = 0
            self.kld_weight = 0

        return e_input

class SoftAttentiveEmbedding(SoftEmbedding):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.pooler is not None, \
                'the pooler was not succesfully assigned.'

    def forward(self, tokens, is_train=False, steps=1):
        seq_length = tokens.size()[-1]
        e_input = super().forward(tokens, is_train, steps)
        pooled_output = self.pooler(e_input, None, None)[0]

        return pooled_output

class SoftAdaptiveEmbeddingDev(SoftEmbedding):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, tokens, is_train=False, steps=1, clf_labels=None):
        """ `training` and `evaluation/prediction` phrases have diff setups. 
        The training batch contains both (specified) batch_size * 2 
        as exact mini-batch during training; while evaluation batch is unmodified."""

        batch_size, seq_length = tokens.shape
        e_source = self.orig_embeds(tokens)
        e_prompt = self.prompt_embeds.unsqueeze(0) 

        if is_train:
            # reparam (1 N H) --> (1 N D) --> (B N H)
            mean = self.hidden2mean(e_prompt)
            logv = self.hidden2logv(e_prompt)
            std = torch.exp(0.5*logv)
            z = torch.cat([
                mean + torch.randn(mean.shape, device=mean.device)* (1-i) * std \
                    for i in clf_labels
            ], 0)
            e = self.latent2hidden(z) 

            # [reshape] All done

            # [concat]
            ## (B N H) & (B L H)
            e_input = torch.cat([e, e_source], 1)

            # [compute loss]
            loss = kl_loss(logv.view(-1, self.latent_size),
                           mean.view(-1, self.latent_size)) 
            self.kld_loss = loss
            self.kld_weight = kl_weight(**self.kld_kwargs, steps=steps)

        else: 
            # reparam (1 N H) --> (1 N D) --> (K N H)
            mean = self.hidden2mean(e_prompt)
            logv = self.hidden2logv(e_prompt)
            std = torch.exp(0.5*logv)
            z = torch.cat([mean + i * std for i in self.std_list], 0)
            e = self.latent2hidden(z)

            # [reshape]
            e = e.repeat_interleave(e_source.size(0), dim=0)
            e_source = e_source.repeat(len(self.std_list), 1, 1)

            # [concat]
            e_input = torch.cat([e, e_source], 1)

            self.kld_loss = 0
            self.kld_weight = 0

        return e_input

