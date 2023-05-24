import torch
import inspect
from typing import Optional, Tuple, Union, Dict, Any
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
            used_vocab_idx: str = None, 
            pooler: Optional[object] = None,
            **kwargs
        ):
        super(SoftEmbedding, self).__init__()
        # [NOTE] If prepending <s> is needed
        self.n_prompts = n_prompts
        self.orig_embeds = wte

        if initialize_from_vocab:
            if isinstance(used_vocab_idx, list):
                used_vocab_idx = used_vocab_idx*self.n_prompts
                used_vocab_idx = used_vocab_idx[:self.n_prompts]
            else:
                vocab_size = wte.weight.size(-1) - 1
                used_vocab_idx = list(
                        range(vocab_size-self.n_prompts, vocab_size, 1)
                )
            self.prompt_embeds = nn.Parameter(
                    self.orig_embeds.weight[used_vocab_idx].clone().detach()
            )
        else:
            self.prompt_embeds = nn.Parameter(
                    torch.randn((self.n_prompts, hidden_size), device=wte.weight.device)
            )

        if kwargs.pop("has_compressed_layer"):
            self.hidden2mean = nn.Linear(2*latent_size, latent_size, bias=False)
            self.hidden2logv = nn.Linear(2*latent_size, latent_size, bias=False)
            self.latent2hidden = nn.Linear(latent_size*2, hidden_size, bias=False)

            self.downproject = nn.Linear(hidden_size, latent_size*2, bias=False)
            self.upproject = nn.Linear(latent_size, latent_size*2, bias=False)
        else:
            self.hidden2mean = nn.Linear(hidden_size, latent_size, bias=False)
            self.hidden2logv = nn.Linear(hidden_size, latent_size, bias=False)
            self.latent2hidden = nn.Linear(latent_size, hidden_size, bias=False)
            self.upproject = None
            self.downproject = None

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.pooler = pooler
        self.kld_kwargs = kwargs

    def forward(self, tokens, is_train=False, steps=1, clf_labels=None):
        """ `training` and `evaluation/prediction` phrases have diff setups. 
        The training batch contains both (specified) batch_size * 2 
        as exact mini-batch during training; while evaluation batch is unmodified."""

        batch_size, seq_length = tokens.shape
        e_source = self.orig_embeds(tokens)
        e_prompt = self.prompt_embeds.unsqueeze(0) 
        if self.downproject is not None:
            e_prompt = self.downproject(e_prompt)

        if is_train:
            # (1, Np, H)-->(1, Np, Hz)-->(1*2, Np, H)
            mean = self.hidden2mean(e_prompt)
            logv = self.hidden2logv(e_prompt)
            std = torch.exp(0.5*logv)
            r = torch.randn(mean.shape, device=e_source.device)
            z = torch.cat([
                mean+torch.randn(mean.shape, device=mean.device)*(1-i)*std \
                    for i in clf_labels
            ], 0)
            if self.upproject is not None:
                z = self.upproject(z)
            e = self.latent2hidden(z) 

            # [reshape] All done

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
            if self.upproject is not None:
                z = self.upproject(z)
            e = self.latent2hidden(z)

            # [reshape]
            e = e.repeat_interleave(batch_size, dim=0)
            e_source = e_source.repeat(len(self.std_list), 1, 1)

            self.kld_loss = 0
            self.kld_weight = 0

        # [concat]
        e_input = torch.cat([e, e_source], 1)

        return e_input

class SoftAdaptiveEmbedding(SoftEmbedding):

    def forward(self, tokens, is_train=False, steps=1, clf_labels=None):
        batch_size, seq_length = tokens.shape
        e_source = self.orig_embeds(tokens)
        e_prompt = self.prompt_embeds.unsqueeze(0) 

        # [pooling] mean or max
        e_pooled = torch.mean(e_source, dim=1).unsqueeze(1)
        # e_pooled = torch.max(e_source, dim=1).values.unsqueeze(1)
        if self.downproject is not None:
            e_pooled = self.downproject(e_pooled)

        if is_train: 
            mean = self.hidden2mean(e_pooled)
            logv = self.hidden2logv(e_pooled)
            std = torch.exp(0.5*logv)
            r = torch.randn(mean.shape, device=e_source.device)
            r[(clf_labels==1)] = 0
            z  = mean + r * std
            if self.upproject is not None:
                z = self.upproject(z)
            e = self.latent2hidden(z) 

            # [reshape] All done
            # (1, N, H) --> (B, N, H)
            e_prompt = e_prompt.repeat(batch_size, 1, 1)
            # (B, N, H) + (B, 1, H) = (B, N, H)
            e = e_prompt + e

            # compute loss
            loss = kl_loss(logv.view(-1, self.latent_size),
                           mean.view(-1, self.latent_size))
            self.kld_loss = loss
            self.kld_weight = kl_weight(**self.kld_kwargs, steps=steps)

        else: 
            mean = self.hidden2mean(e_pooled)
            logv = self.hidden2logv(e_pooled)
            std = torch.exp(0.5*logv)
            z = torch.cat([mean+i*std for i in self.std_list], 0)
            if self.upproject is not None:
                z = self.upproject(z)
            e = self.latent2hidden(z) 

            # [reshape]
            # (B*std, 1, H)
            # (1, N, H) --> (B*std, N, H)
            e_prompt = e_prompt.repeat(len(self.std_list)*batch_size, 1, 1)
            e = e_prompt + e
            e_source = e_source.repeat(len(self.std_list), 1, 1)

            # compute loss
            self.kld_loss = 0
            self.kld_weight = 0

        # Concat z to original embeddings
        e_input = torch.cat([e, e_source], 1)

        return e_input

class SoftEmbeddingWithPooler(SoftEmbedding):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.pooler is not None, \
                'the pooler was not succesfully assigned.'

    def forward(self, tokens, is_train=False, steps=1, clf_labels=None):
        seq_length = tokens.size()[-1]
        e_input = super().forward(tokens, is_train, steps, clf_labels)
        pooled_output = self.pooler(e_input, None, None)[0]

        return pooled_output

class SoftEmbeddingBasic(SoftEmbedding):

    def forward(self, tokens, is_train=False, steps=1, clf_labels=None):
        batch_size, seq_length = tokens.shape
        e_source = self.orig_embeds(tokens)
        e_prompt = self.prompt_embeds.unsqueeze(0) 
        if self.downproject is not None:
            e_repr = self.downproject(e_source)

        if is_train:
            # (B, L, H)
            mean = self.hidden2mean(e_repr)
            logv = self.hidden2logv(e_repr)
            std = torch.exp(0.5*logv)
            r = torch.randn(mean.shape, device=e_source.device)
            r[(clf_labels == 1)] = 0
            z = mean + r 
            if self.upproject is not None:
                z = self.upproject(z)
            e = self.latent2hidden(z) 

            # [reshape] 
            # (1, N, H) --> (B, N, H)
            e_prompt = e_prompt.repeat(batch_size, 1, 1)
            # (B, N, H) + (B, 1, H) = (B, N, H)
            e = e_prompt + torch.mean(e, dim=1).unsqueeze(1)

            # compute loss
            loss = kl_loss(logv.view(-1, self.latent_size),
                           mean.view(-1, self.latent_size)) 
            self.kld_loss = loss
            self.kld_weight = kl_weight(**self.kld_kwargs, steps=steps)

        else: 
            mean = self.hidden2mean(e_repr)
            logv = self.hidden2logv(e_repr)
            std = torch.exp(0.5*logv)
            z = torch.cat([mean+std*i for i in self.std_list], 0)
            if self.upproject is not None:
                z = self.upproject(z)
            e = self.latent2hidden(z)

            # [reshape]
            # (1, N, H) --> (B*std, N, H)
            e_prompt = e_prompt.repeat(len(self.std_list)*batch_size, 1, 1)
            # (B*std, N, H) + (B*std, 1, H) = (B*std, N, H)
            e = e_prompt + torch.mean(e, dim=1).unsqueeze(1)

            # (B, L, H) --> (B*std, L, H)
            e_source = e_source.repeat(len(self.std_list), 1, 1)

            self.kld_loss = 0
            self.kld_weight = 0

        # [concat]
        e_input = torch.cat([e, e_source], 1)

        return e_input

class SoftEmbeddingForDecoder(SoftEmbedding):

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
        self.latent_size = latent_size

        if initialize_from_vocab:
            if isinstance(initialize_from_vocab, list):
                self.prompt_embeds = nn.Parameter(
                        self.orig_embeds.weight[
                            initialize_from_vocab
                        ].clone().detach()
                )
            else:
                self.prompt_embeds = nn.Parameter(
                        self.orig_embeds.weight[-n_prompts:].clone().detach()
                )
        else:
            self.prompt_embeds = nn.Parameter(
                    torch.randn((self.n_prompts, hidden_size), device=wte.weight.device)
            )

    def forward(self, tokens):
        e_source = self.orig_embeds(tokens)
        e_prompt = self.prompt_embeds.unsqueeze(0) 
        return torch.cat([e_prompt.repeat(e_source.size(0), 1, 1), e_source], 1)

class SoftResidualEmbedding(SoftEmbedding):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_embeds_base = nn.Parameter(
                torch.randn(
                    (self.n_prompts, self.hidden_size), 
                    device=self.orig_embeds.weight.device
                )
        )

    def forward(self, tokens, is_train=False, steps=1, clf_labels=None):
        batch_size, seq_length = tokens.shape
        e_source = self.orig_embeds(tokens)
        e_prompt = self.prompt_embeds.unsqueeze(0) 
        if self.downproject is not None:
            e_prompt = self.downproject(e_prompt)

        if is_train: 
            mean = self.hidden2mean(e_prompt)
            logv = self.hidden2logv(e_prompt)
            std = torch.exp(0.5*logv)
            z = torch.cat([
                mean + torch.randn(mean.shape, device=e_source.device)*(1-i)*std \
                        for i in clf_labels
            ], 0) 
            if self.upproject is not None:
                z = self.upproject(z)
            e = self.latent2hidden(z) 

            if self.prompt_embeds_base is not None:
                e = e + self.prompt_embeds_base.unsqueeze(0) 
            
            # [reshape] All done

            # compute loss
            loss = kl_loss(logv.view(-1, self.latent_size),
                           mean.view(-1, self.latent_size))
            self.kld_loss = loss
            self.kld_weight = kl_weight(**self.kld_kwargs, steps=steps)

        else: 
            mean = self.hidden2mean(e_prompt)
            logv = self.hidden2logv(e_prompt)
            std = torch.exp(0.5*logv)
            z = torch.cat([mean+i*std for i in self.std_list], 0)
            if self.upproject is not None:
                z = self.upproject(z)
            e = self.latent2hidden(z) 

            if self.prompt_embeds_base is not None:
                e = e + self.prompt_embeds_base.unsqueeze(0) 

            # [reshape]
            e = e.repeat_interleave(batch_size, dim=0)
            e_source = e_source.repeat(len(self.std_list), 1, 1)

            # compute loss
            self.kld_loss = 0
            self.kld_weight = 0

        # Concat z to original embeddings
        e_input = torch.cat([e, e_source], 1)

        return e_input
