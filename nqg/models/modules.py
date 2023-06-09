import math
import torch
from typing import Optional, Tuple, Union, Dict, Any, List
from torch import nn
from torch.nn import functional as F
from utils import kl_weight, kl_loss
import copy
from transformers.activations import gelu

class DocRelPrompt(nn.Module):
    def __init__(
            self, 
            wte: nn.Embedding,
            hidden_size: int = 768,
            head_size: int = 64,
            lbl_init_idx: Optional[List] = None,
            init_idx: Optional[List] = None,
        ):
        super(DocRelPrompt, self).__init__()
        self.orig_embeds = wte
        self.lbl_init_idx = lbl_init_idx
        self.init_idx = init_idx
        self.length = len(init_idx) 

        self.prompts = nn.Parameter(torch.randn(
            (len(init_idx), hidden_size), device=wte.weight.device
        ))
        self.label_prompts = nn.Parameter(torch.randn(
            (len(lbl_init_idx), hidden_size), device=wte.weight.device
        ))

        self.adapter = InstanceWisePrompt(
                wte=None,
                hidden_size=hidden_size,
                head_size=head_size
        )
        self.label_adapter = InstanceWisePrompt(
                wte=None,
                hidden_size=hidden_size,
                head_size=head_size
        )

    def set_embeddings(self):
        self.prompts = nn.Parameter(
                self.orig_embeds.weight[self.init_idx].clone().detach()
        )
        self.label_prompts = nn.Parameter(
                self.orig_embeds.weight[self.lbl_init_idx].clone().detach()
        )
        print(f"{self.__class__.__name__} embeddings set: ", self.init_idx, self.lbl_init_idx)

    def forward(self, relevance, input_ids=None, hidden_states_src=None, steps=None):
        # [TODO] Try different way to model relevance
        if hidden_states_src is None:
            self.n_samples = len(relevance) // input_ids.size(0)
            hidden_states_src = self.orig_embeds(input_ids)
        else:
            self.n_samples = len(relevance) // hidden_states_src.size(0)

        relevance = torch.cat([(1-relevance).view(-1, 1), relevance.view(-1, 1)], 1)
        relevance = relevance.to(self.label_prompts.device)
        hidden_states_rel = torch.matmul(relevance, self.label_prompts).unsqueeze(1) 

        lbl_prompts = self.label_adapter(hidden_states_rel, self.prompts)
        doc_prompts = self.adapter(hidden_states_src, self.prompts)

        # [NOTE]
        # Generally, hidden state injection is much more effective
        # However, naive prompt can not help the variaety
        # Best: return torch.cat([doc_prompts, hidden_states_src+hidden_states_rel], 1)
        return torch.cat([lbl_prompts, hidden_states_src+hidden_states_rel], 1)
    
    def expand_mask(self, mask):
        mask = mask.repeat(self.n_samples, 1)
        additional_mask = torch.ones((mask.size(0), self.length), device=mask.device)
        mask = torch.cat([additional_mask, mask], -1)
        return mask

class RelPrompt(nn.Module):
    def __init__(
            self, 
            wte: nn.Embedding,
            hidden_size: int = 768,
            head_size: int = 64,
            pos_init_idx: Optional[List] = None,
            neg_init_idx: Optional[List] = None,
        ):
        super(RelPrompt, self).__init__()
        self.orig_embeds = wte
        self.pos_init_idx = pos_init_idx
        self.neg_init_idx = neg_init_idx
        # self.length = len(init_idx) 

        self.pos_prompts = nn.Parameter(torch.randn(
            (len(pos_init_idx), hidden_size), device=wte.weight.device
        ))
        self.neg_prompts = nn.Parameter(torch.randn(
            (len(neg_init_idx), hidden_size), device=wte.weight.device
        ))

        self.adapter = InstanceWisePrompt(
                wte=None,
                hidden_size=hidden_size,
                head_size=head_size,
                pooling='max'
        )

    def set_embeddings(self):
        self.pos_prompts = nn.Parameter(
                self.orig_embeds.weight[self.pos_init_idx].clone().detach()
        )
        self.neg_prompts = nn.Parameter(
                self.orig_embeds.weight[self.neg_init_idx].clone().detach()
        )
        print(f"{self.__class__.__name__} embeddings set: ", \
                self.pos_init_idx, self.neg_init_idx)

    def forward(self, relevance, input_ids=None, hidden_states_src=None, steps=None):
        batch_size = hidden_states_src.size(0)
        if hidden_states_src is None:
            self.n_samples = len(relevance) // input_ids.size(0)
            hidden_states_src = self.orig_embeds(input_ids)
        else:
            self.n_samples = len(relevance) // hidden_states_src.size(0)

        # [relevance conditioned]
        ## B N H
        relevance = relevance.to(self.pos_prompts.device)
        pos_prompts = self.pos_prompts.unsqueeze(0).repeat(batch_size, 1, 1)
        neg_prompts = self.neg_prompts.unsqueeze(0).repeat(batch_size, 1, 1)
        rel_prompts = (relevance.view(-1, 1, 1) * pos_prompts + \
                       (1-relevance).view(-1, 1, 1) * neg_prompts) / 2

        # [NOTE]
        ## Setting 1: add hidden_states_src directly to encoder_outputs
        rel_hidden_states_src = self.adapter(rel_prompts, hidden_states_src) 
        return rel_hidden_states_src

    def expand_mask(self, mask):
        mask = mask.repeat(self.n_samples, 1)
        additional_mask = torch.ones((mask.size(0), self.length), device=mask.device)
        mask = torch.cat([additional_mask, mask], -1)
        return mask


class InstanceWisePrompt(nn.Module):
    """
    Note that instancewise prompt uses the "prompt" as base.
    And uses the "hidden" as instance-aware features.
    """
    def __init__(
            self, 
            wte: nn.Embedding, 
            hidden_size: int = 768,
            head_size: int = 64,
            length: int = 1,
            init_idx: Optional[List] = None,
            pooling: str = 'mean',
        ):
        super(InstanceWisePrompt, self).__init__()
        self.orig_embeds = wte
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.length = length
        self.q_proj = nn.Linear(hidden_size, head_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, head_size, bias=True)
        self.pooling = pooling

        if init_idx is not None:
            self.init_idx = (init_idx*length)[:length]
            self.prompt_embeds = nn.Parameter(
                    torch.randn((length, hidden_size), device=wte.weight.device)
            )

    def set_embeddings(self):
        self.prompt_embeds = nn.Parameter(
                self.orig_embeds.weight[self.init_idx].clone().detach()
        )
        print("Set embeddings to", self.init_idx)

    def forward(self, hidden_states=None, base=None, input_ids=None):
        """
        instance-aware/hidden_states: B L H  (input_ids B L)
        base: N H 
        """
        if input_ids is not None:
            hidden_states = self.orig_embeds(input_ids)
        if base is None:
            base = self.prompt_embeds

        if base.size(0) == hidden_states.size(0):
            V = base
        else:
            V = base.unsqueeze(0).repeat(hidden_states.size(0), 1, 1)

        Q = self.q_proj(base) 
        K = self.k_proj(hidden_states) 
        K = torch.transpose(K, 1, 2)
        # scores: B N L
        scores = torch.matmul(Q, K)   
        # scores: B N
        if self.pooling == 'mean':
            scores = torch.mean(scores / math.sqrt(self.head_size), dim=-1)
        else:
            scores = torch.max(scores / math.sqrt(self.head_size), dim=-1).values
        probs = torch.sigmoid(scores)
        # B N H * B N 1
        scores = torch.mean(scores / math.sqrt(self.head_size), dim=-1)
        iw_prompts = torch.mul(V, probs.unsqueeze(-1))
        return iw_prompts
    
    def expand_mask(self, mask):
        additional_mask = torch.ones((mask.size(0), self.length), device=mask.device)
        mask = torch.cat([additional_mask, mask], 1)
        return mask

class VAE(nn.Module):
    def __init__(
            self,
            hidden_size: int = 768, 
            latent_size: int = 128, 
            length: int = 1,
            prefix_length: int = 0,
            device: Optional[str] = 'cpu',
            **kwargs
        ):
        super(EncDecVAE, self).__init__()
        self.orig_embeds = wte

        cond_size = hidden_size
        self.downproject = nn.Linear(cond_size+hidden_size, latent_size*2, bias=False)
        self.hidden2mean = nn.Linear(latent_size*2, latent_size, bias=False)
        self.hidden2logv = nn.Linear(latent_size*2, latent_size, bias=False)
        self.upproject = nn.Linear(cond_size+latent_size, latent_size*2, bias=False)
        self.latent2hidden = nn.Linear(latent_size*2, hidden_size, bias=False)

        self.downproject_pri = nn.Linear(hidden_size, latent_size*2, bias=False)
        self.hidden2mean_pri = nn.Linear(latent_size*2, latent_size, bias=False)
        self.hidden2logv_pri = nn.Linear(latent_size*2, latent_size, bias=False)

        # attr
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.length = length
        self.prefix_length = prefix_length
        self.device = device
        self.kld_kwargs = kwargs.pop('kld_kwargs')

    def forward(self, enc_last_hidden_state, relevance=None, steps=None):
        h_prompt = enc_last_hidden_state[:, :self.prefix_length]
        h_source = enc_last_hidden_state[:, self.prefix_length:]

        sent_source = torch.mean(h_source, dim=1).unsqueeze(1)
        sent_prompt = torch.mean(h_prompt, dim=1).unsqueeze(1)
        # if self.pooling == 'max':
        #     h_sent = torch.max(h_source, dim=1).values.unsqueeze(1)

        # vae (encoder)
        ## label encoding 
        condition = torch.cat([(1-relevance).view(-1, 1), relevance.view(-1, 1)], 1)
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
            learnable_prior: bool = False,
            **kwargs
        ):
        super(EncDecCVAE, self).__init__()
        self.orig_embeds = wte
        concat = hidden_size if pooling != 'attentive' else 0

        if has_compressed_layer:
            self.down_proj = nn.Linear(concat+hidden_size, latent_size*2, bias=False)
            self.h2mean = nn.Linear(latent_size*2, latent_size, bias=False)
            self.h2logv = nn.Linear(latent_size*2, latent_size, bias=False)
            self.up_proj = nn.Linear(hidden_size+latent_size, latent_size*2, bias=False)
            self.l2h = nn.Linear(latent_size*2, hidden_size, bias=False)
            if learnable_prior:
                self.down_proj_pri = nn.Linear(concat+hidden_size, latent_size*2, bias=False)
                self.h2mean_pri = nn.Linear(latent_size*2, latent_size, bias=False)
                self.h2logv_pri = nn.Linear(latent_size*2, latent_size, bias=False)
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

        self.learnable_prior = learnable_prior

    def set_embeddings(self):
        self.label_embeds = nn.Parameter(
                self.orig_embeds.weight[self.init_idx].clone().detach()
        )
        print("Set embeddings to", self.init_idx)

    def forward(self, enc_last_hidden_state, relevance=None, steps=None):
        h_prompt = enc_last_hidden_state[:, :self.prefix_length]
        h_source = enc_last_hidden_state[:, self.prefix_length:]
        ## label encoding if needed
        condition = torch.cat(
                [1-relevance.view(-1, 1), relevance.view(-1, 1)], 1
        )
        h_cond = torch.matmul(condition, self.label_embeds).unsqueeze(1)

        # vae (encoder)
        ## condition 1
        if self.pooling == 'attentive':
            h_post = AttentivePooling(enc_last_hidden_state, 
                    self.label_embeds, condition).unsqueeze(1)
            h_pri = AttentivePooling(h_source, 
                    self.label_embeds, condition).unsqueeze(1)
        else:
            # B N H / B L H --> B 1 H 
            if self.pooling == 'mean':
                h_aspect = torch.mean(h_aspect, dim=1).unsqueeze(1)
                h_source = torch.mean(h_source, dim=1).unsqueeze(1)
            if self.pooling == 'max':
                h_aspect = torch.max(h_aspect, dim=1).values.unsqueeze(1)
                h_source = torch.max(h_source, dim=1).values.unsqueeze(1)

            h_post = torch.cat([h_aspect, h_source+h_cond], -1)
            h_pri = torch.cat([h_aspect, h_cond], -1)

        if self.down_proj is not None:
            h_post = self.down_proj(h_post)

        mean = self.h2mean(h_post)
        logv = self.h2logv(h_post)
        std = logv.mul(0.5).exp()
        r = torch.zeros_like(std).normal_() if steps else 0
        z = mean + r * std

        ## vae (decoder)
        # label encoding 
        z = torch.cat((z, h_source+h_cond), -1)
        if self.up_proj is not None:
            z = self.up_proj(z)
        h_post = self.l2h(z)

        # compute the prior
        if self.learnable_prior:
            h_pri = self.down_proj_pri(h_pri)
            mean_pri = self.h2mean_pri(h_pri).view(-1, self.latent_size)
            logv_pri = self.h2logv_pri(h_pri).view(-1, self.latent_size)
        else:
            mean_pri = None
            logv_pri = None


        # compuate loss
        loss = kl_loss(logv.view(-1, self.latent_size), 
                       mean.view(-1, self.latent_size),
                       logv_pri, mean_pri
        )
        weight = kl_weight(**self.kld_kwargs, steps=steps)

        return h_post, loss*weight

def AttentivePooling(hidden_x, label_embeds, condition):
    """
    hidden_x: B L H
    hidden_c: 2 H 
    scores: B L 2 * B 1 2 -> B L 2 --> (average) B L
    output_x: B L H * B L 1 --> B L H 
    """
    scores = torch.matmul(hidden_x, label_embeds.transpose(1, 0))   
    # scores = torch.mean(scores / math.sqrt(label_embeds.size(-1)), dim=-1)
    probs = torch.sigmoid(scores)
    probs = torch.mul(probs, condition.unsqueeze(1)).mean(-1)
    hidden_x_attn = torch.mul(hidden_x, probs.unsqueeze(-1))
    return hidden_x_attn.mean(1)

