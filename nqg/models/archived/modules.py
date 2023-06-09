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
        self.length = len(init_idx) + len(init_idx)

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

    def forward(self, relevance, input_ids, steps=None):
        # [TODO] Try different way to model relevance
        self.n_samples = len(relevance) // input_ids.size(0)
        # input_ids = input_ids.expand(relevance.size(0), 1, 1)
        hidden_states_src = self.orig_embeds(input_ids)
        relevance = torch.cat([(1-relevance).view(-1, 1), relevance.view(-1, 1)], 1)
        hidden_states_rel = torch.matmul(relevance, self.label_prompts).unsqueeze(1) 

        lbl_prompts = self.label_adapter(hidden_states_rel, self.prompts)
        doc_prompts = self.adapter(hidden_states_src, self.prompts)

        return torch.cat([lbl_prompts, doc_prompts, hidden_states_src], 1)
    
    def expand_mask(self, mask):
        mask = mask.repeat(self.n_samples, 1)
        additional_mask = torch.ones((mask.size(0), self.length), device=mask.device)
        mask = torch.cat([additional_mask, mask], -1)
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

    def forward(self, hidden_states=None, prompt_embeds=None, input_ids=None):
        """
        hidden_states: B L H  (input_ids B L)
        prompt_embeds: N H 
        """
        if input_ids is not None:
            hidden_states = self.orig_embeds(input_ids)
        if prompt_embeds is None:
            prompt_embeds = self.prompt_embeds

        V = prompt_embeds.unsqueeze(0).repeat(hidden_states.size(0), 1, 1)
        Q = self.q_proj(prompt_embeds) 
        K = self.k_proj(hidden_states) 
        K = torch.transpose(K, 1, 2)
        # scores: B N L
        scores = torch.matmul(Q, K)   
        # scores: B N
        scores = torch.mean(scores / math.sqrt(self.head_size), dim=-1)
        probs = torch.sigmoid(scores)
        # B N H * B N 1
        scores = torch.mean(scores / math.sqrt(self.head_size), dim=-1)
        iw_prompts = torch.mul(V, probs.unsqueeze(-1))
        return iw_prompts
    
    def expand_mask(self, mask):
        additional_mask = torch.ones((mask.size(0), self.length), device=mask.device)
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

class SoftAdaptiveEmbedding(nn.Module):

    def set_embeddings(self):
        self.prompt_embeds = nn.Parameter(
                self.orig_embeds.weight[self.init_idx].clone().detach()
        )
        print("Set embeddings to", self.init_idx)

    def __init__(
            self,
            wte: nn.Embedding,
            hidden_size: int = 768, 
            latent_size: int = 128, 
            n_prompts: int = 1,
            initialize_from_vocab: bool = False, 
            used_prompt_idx: List[int] = None, 
            adaptive_pooling: Optional[str] = None,
            has_compressed_layer: bool = False,
            **kwargs
        ):
        super(SoftAdaptiveEmbedding, self).__init__()
        self.n_prompts = n_prompts
        self.orig_embeds = wte

        if initialize_from_vocab:
            if isinstance(used_prompt_idx, list):
                used_prompt_idx = used_prompt_idx*self.n_prompts
                self.init_idx = used_prompt_idx[:self.n_prompts]
            else:
                vocab_size = wte.weight.size(-1) - 1
                self.init_idx = list(range(vocab_size-self.n_prompts, vocab_size, 1))
        else:
            self.prompt_embeds = nn.Parameter(
                    torch.randn((self.n_prompts, hidden_size), device=wte.weight.device)
            )

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.adaptive_pooling = adaptive_pooling
        self.pooler = kwargs.pop('pooler')
        self.kld_kwargs = kwargs
        self.has_compressed_layer = has_compressed_layer
        self.post_init()

    def post_init(self):
        super().post_init()
        self.downproject = nn.Linear(
                self.hidden_size+2, self.latent_size*2,
                bias=False
        )
        self.upproject = nn.Linear(
                self.latent_size+self.hidden_size+2, self.latent_size*2, 
                bias=False
        )

    def forward(self, tokens, is_train=False, steps=1, clf_labels=None):
        batch_size, seq_length = tokens.shape
        e_source = self.orig_embeds(tokens)
        e_prompt = self.prompt_embeds.unsqueeze(0) 

        # [pooling] mean or max
        if self.adaptive_pooling == 'mean':
            e_pooled = torch.mean(e_source, dim=1).unsqueeze(1)
        elif self.adaptive_pooling == 'max':
            e_pooled = torch.max(e_source, dim=1).values.unsqueeze(1)

        if is_train: 
            # add the contditional tokens
            # B 2
            cond = torch.cat([(1-clf_labels).view(-1, 1), clf_labels.view(-1, 1)], 1)
            cond = cond.unsqueeze(1)
            # B 1 2, B 1 H
            h_pooled = torch.cat([e_pooled, cond], -1)
            if self.downproject is not None:
                h_pooled = self.downproject(h_pooled)

            mean = self.hidden2mean(h_pooled)
            logv = self.hidden2logv(h_pooled)
            std = torch.exp(0.5*logv)
            r = torch.randn(mean.shape, device=e_source.device)
            z = mean + r * std 

            z = torch.cat([z, e_pooled, cond], -1)
            if self.upproject is not None:
                z = self.upproject(z)
            e = self.latent2hidden(z) 

            # [reshape] All done
            # (1, N, H) --> (B, N, H)
            e_prompt = e_prompt.repeat(batch_size, 1, 1)
            # (B, N, H) + (B, 1, H) = (B, N, H)
            e = e_prompt + e 

            # compute loss
            self.kld_loss = kl_loss(logv.view(-1, self.latent_size),
                           mean.view(-1, self.latent_size))
            self.kld_weight = kl_weight(**self.kld_kwargs, steps=steps)
        else: 
            clf_labels = torch.range(0, len(self.std_list)-1, 1)/(len(self.std_list)-1)
            # N 2
            cond = torch.cat([(1-clf_labels).view(-1, 1), clf_labels.view(-1, 1)], 1)
            cond = cond.unsqueeze(1).to(e_pooled.device)
            # BN 2
            cond = cond.repeat_interleave(batch_size, dim=0)
            # BN 1 H
            e_pooled = e_pooled.repeat(len(self.std_list), 1, 1)
            # BN 1 H + BN 1 2
            h_pooled = torch.cat([e_pooled, cond], -1)
            if self.downproject is not None:
                h_pooled = self.downproject(h_pooled)

            mean = self.hidden2mean(h_pooled)
            z = mean 

            z = torch.cat([z, e_pooled, cond], -1)
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
