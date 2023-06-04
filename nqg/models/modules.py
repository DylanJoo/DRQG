import math
import torch
from typing import Optional, Tuple, Union, Dict, Any, List
from torch import nn
from torch.nn import functional as F
from utils import kl_weight, kl_loss
import copy

class RelevancePrompt(nn.Module):
    def __init__(
            self, 
            wte: nn.Embedding,
            hidden_size: int = 768,
            init_idx: Optional[List] = None,
            length: int = 1,
        ):
        super(RelevancePrompt, self).__init__()
        self.orig_embeds = wte
        self.init_idx = init_idx
        self.length = length
        self.prompt_embeds = nn.Parameter(
                torch.randn((len(init_idx), hidden_size), device=wte.weight.device)
        )
    def set_embeddings(self):
        self.prompt_embeds = nn.Parameter(
                self.orig_embeds.weight[self.init_idx].clone().detach()
        )
        print("Set embeddings to", self.init_idx)

    def forward(self, input_ids, relevance=None, steps=None):
        # B L / NB L --> B L H / NB L H
        n_samples = len(relevance) // input_ids.size(0)
        h_source = self.orig_embeds(input_ids)
        h_source = h_source.repeat(n_samples, 1, 1)

        # B / NB --> B 2/ NB 2--> B 1 H / NB 1 H
        relevance = torch.cat([(1-relevance).view(-1, 1), relevance.view(-1, 1)], 1)
        h_relevance = torch.matmul(relevance, self.prompt_embeds).unsqueeze(1) 

        return torch.cat([h_relevance, h_source], 1)
    
    def expand_mask(self, mask, relevance):
        additional_mask = torch.ones(
                (mask.size(0), self.length),
                device=mask.device
        )
        mask = torch.cat([additional_mask, mask], 1)
        mask = mask.expand(relevance.size(0), mask.size(1))
        return mask

class InstanceWisePrompt(nn.Module):
    def __init__(
            self, 
            wte: nn.Embedding, 
            hidden_size: int = 768,
            head_size: int = 64,
            length: int = 1,
            init_idx: Optional[List] = None,
        ):
        super(InstanceWisePrompt, self).__init__()
        self.orig_embeds = wte
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.length = length
        self.q_proj = nn.Linear(hidden_size, head_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, head_size, bias=True)

        self.prompt_embeds = nn.Parameter(
                torch.randn((length, hidden_size), device=wte.weight.device)
        )
        self.init_idx = (init_idx*length)[:length]

    def set_embeddings(self):
        self.prompt_embeds = nn.Parameter(
                self.orig_embeds.weight[self.init_idx].clone().detach()
        )
        print("Set embeddings to", self.init_idx)

    def forward(self, hidden_states=None, tokens=None):
        """
        prompr_embeds: N H --> B N H
        hidden_states: B 1 H 
        """
        if tokens is not None:
            hidden_states = self.orig_embeds(tokens)
        V = self.prompt_embeds.unsqueeze(0).repeat(hidden_states.size(0), 1, 1)
        Q = self.q_proj(self.prompt_embeds) 
        K = self.k_proj(hidden_states) 
        K = torch.transpose(K, 1, 2)
        scores = torch.matmul(Q, K)   
        scores = torch.mean(scores / math.sqrt(self.head_size), dim=-1)
        probs = torch.sigmoid(scores)
        iw_prompts = torch.mul(V, probs.unsqueeze(-1))
        return iw_prompts
    
    def expand_mask(self, mask):
        additional_mask = torch.ones(
                (mask.size(0), self.length),
                device=mask.device
        )
        mask = torch.cat([additional_mask, mask], 1)
        return mask


class EncDecVAE(nn.Module):
    def __init__(
            self,
            wte: nn.Embedding, 
            hidden_size: int = 768, 
            latent_size: int = 128, 
            length: int = 1,
            prefix_length: int = 0,
            pooling: Optional[str] = 'mean',
            device: Optional[str] = 'cpu',
            has_compressed_layer: bool = True,
            init_idx: Optional[List[int]] = None, 
            **kwargs
        ):
        super(EncDecVAE, self).__init__()
        """ 
        This class is used for traditional CVAE. `length` indicates the number of 
        encoder prompts (at the static embedding layer)
        """
        self.orig_embeds = wte

        if has_compressed_layer:
            self.downproject = nn.Linear(hidden_size+hidden_size, latent_size*2, bias=False)
            self.hidden2mean = nn.Linear(latent_size*2, latent_size, bias=False)
            self.hidden2logv = nn.Linear(latent_size*2, latent_size, bias=False)
            self.upproject = nn.Linear(hidden_size+latent_size, latent_size*2, bias=False)
            self.latent2hidden = nn.Linear(latent_size*2, hidden_size, bias=False)

            self.downproject_pri = nn.Linear(hidden_size, latent_size*2, bias=False)
            self.hidden2mean_pri = nn.Linear(latent_size*2, latent_size, bias=False)
            self.hidden2logv_pri = nn.Linear(latent_size*2, latent_size, bias=False)
        else:
            self.hidden2mean = nn.Linear(hidden_size+hidden_size, latent_size, bias=False)
            self.hidden2logv = nn.Linear(hidden_size+hidden_size, latent_size, bias=False)
            self.latent2hidden = nn.Linear(latent_size, hidden_size, bias=False)

        # attr
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.length = length
        self.prefix_length = prefix_length
        self.kld_kwargs = kwargs.pop('kld_kwargs')
        self.pooling = pooling
        self.device = device
        self.init_idx = (init_idx*length)[:length]

        self.label_embeds = nn.Parameter(torch.randn((2, hidden_size), device=device))

    def set_embeddings(self):
        self.label_embeds = nn.Parameter(
                self.orig_embeds.weight[self.init_idx].clone().detach()
        )
        print("Set embeddings to", self.init_idx)

    def forward(self, enc_last_hidden_state, relevance=None, steps=None):
        h_prompt = enc_last_hidden_state[:, :self.prefix_length]
        h_source = enc_last_hidden_state[:, self.prefix_length:]
        if self.pooling == 'mean':
            h_sent = torch.mean(h_source, dim=1).unsqueeze(1)
        if self.pooling == 'max':
            h_sent = torch.max(h_source, dim=1).values.unsqueeze(1)

        # vae (encoder)
        ## label encoding 
        condition = torch.cat([(1-relevance).view(-1, 1), relevance.view(-1, 1)], 1)
        h_cond = torch.matmul(condition, self.label_embeds).unsqueeze(1) # B 1 H
        h_enc = torch.cat((h_sent, h_cond), -1) # B 1 H / B L H

        if self.downproject is not None:
            h_enc = self.downproject(h_enc)
            h_cond_pri = self.downproject_pri(h_cond)
        mean = self.hidden2mean(h_enc)
        logv = self.hidden2mean(h_enc)
        std = torch.exp(0.5*logv)
        r = torch.rand(mean.shape, device=mean.device) 
        if steps is None:
            r = 0
        z = mean + r * std

        # vae (decoder)
        ## label encoding 
        z = torch.cat((z, h_cond), -1)

        if self.upproject is not None:
            z = self.upproject(z)
        h_label = self.latent2hidden(z)

        # Compute the prior
        logv_pri = self.hidden2logv_pri(h_cond_pri)
        mean_pri = self.hidden2mean_pri(h_cond_pri)

        # compuate loss
        loss = kl_loss(
                logv.view(-1, self.latent_size),
                mean.view(-1, self.latent_size),
                logv_pri.view(-1, self.latent_size),
                mean_pri.view(-1, self.latent_size)
        ) 
        weight = kl_weight(**self.kld_kwargs, steps=steps)

        return h_label, loss*weight

class EncDecCVAE(nn.Module):
    def __init__(
            self,
            wte: nn.Embedding,
            hidden_size: int = 768, 
            latent_size: int = 128, 
            prefix_length: int = 0,
            pooling: Optional[str] = 'mean',
            device: Optional[str] = 'cpu',
            init_idx: Optional[List] = None,
            has_compressed_layer: bool = True,
            **kwargs
        ):
        super(EncDecCVAE, self).__init__()
        self.orig_embeds = wte

        if has_compressed_layer:
            times=2
            self.down_proj = nn.Linear(hidden_size+hidden_size, latent_size*2, bias=False)
            self.down_proj_pri = nn.Linear(hidden_size, latent_size*2, bias=False)
            self.h2mean = nn.Linear(latent_size*2, latent_size, bias=False)
            self.h2logv = nn.Linear(latent_size*2, latent_size, bias=False)
            self.h2mean_pri = nn.Linear(latent_size*2, latent_size, bias=False)
            self.h2logv_pri = nn.Linear(latent_size*2, latent_size, bias=False)
            self.up_proj = nn.Linear(hidden_size+latent_size, latent_size*2, bias=False)
            self.l2h = nn.Linear(latent_size*2, hidden_size, bias=False)
        else:
            self.down_proj = None
            self.h2mean = None
            self.h2logv = None
            self.up_proj = None
            self.l2h = None

        # attr
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.prefix_length = prefix_length
        self.kld_kwargs = kwargs.pop('kld_kwargs')
        self.pooling = pooling
        self.device = device
        self.init_idx = init_idx
        self.label_embeds = nn.Parameter(torch.randn((2, hidden_size), device=device))

    def set_embeddings(self):
        self.label_embeds = nn.Parameter(
                self.orig_embeds.weight[self.init_idx].clone().detach()
        )
        print("Set embeddings to", self.init_idx)

    def forward(self, enc_last_hidden_state, relevance=None, steps=None):
        """
        Condition on feature concatenation
        """
        h_aspect = enc_last_hidden_state[:, :self.prefix_length]
        h_source = enc_last_hidden_state[:, self.prefix_length:]
        # B N H / B L H --> B 1 H 
        if self.pooling == 'mean':
            h_aspect = torch.mean(h_aspect, dim=1).unsqueeze(1)
            h_source = torch.mean(h_source, dim=1).unsqueeze(1)
        if self.pooling == 'max':
            h_aspect = torch.max(h_aspect, dim=1).values.unsqueeze(1)
            h_source = torch.max(h_source, dim=1).values.unsqueeze(1)

        # vae (encoder)
        ## condition 1
        ## label encoding 
        condition = torch.cat(
                [1-relevance.view(-1, 1), relevance.view(-1, 1)], 1
        )
        h_cond = torch.matmul(condition, self.label_embeds).unsqueeze(1)
        h_post = torch.cat([h_aspect+h_source, h_cond], -1)
        h_pri = h_cond 
        # h_pri = F.normalize(h_cond, dim=-1)

        if self.down_proj is not None:
            h_post = self.down_proj(h_post)
            h_pri = self.down_proj_pri(h_pri)
        mean = self.h2mean(h_post)
        logv = self.h2logv(h_post)
        std = logv.mul(0.5).exp()
        r = torch.zeros_like(std).normal_() if steps else 0
        z = mean + r * std

        ## vae (decoder)
        # label encoding 
        z = torch.cat((z, h_cond), -1)
        if self.up_proj is not None:
            z = self.up_proj(z)
        h_post = self.l2h(z)

        # compute the prior
        mean_pri = self.h2mean_pri(h_pri)
        logv_pri = self.h2logv_pri(h_pri)

        # compuate loss
        loss = kl_loss(logv.view(-1, self.latent_size), 
                       mean.view(-1, self.latent_size),
                       logv_pri.view(-1, self.latent_size), 
                       mean_pri.view(-1, self.latent_size)
        )
        weight = kl_weight(**self.kld_kwargs, steps=steps)

        return h_post, loss*weight
