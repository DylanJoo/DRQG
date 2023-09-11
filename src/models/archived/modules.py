import math
import torch
from typing import Optional, Tuple, Union, Dict, Any, List
from torch import nn
from torch.nn import functional as F
from utils import kl_weight, kl_loss
import copy

"""
AttentivePrompt
    Note that instancewise prompt uses the "prompt" as base.
    And uses the "hidden" as instance-aware features.
"""
class AttentivePrompt(nn.Module):
    def __init__(
            self, 
            wte: nn.Embedding, 
            hidden_size: int = 768,
            head_size: int = 64,
            length: int = 1,
            init_idx: Optional[List] = None,
            pooling: str = 'mean',
            activation: str = 'sigmoid',
        ):
        super(AttentivePrompt, self).__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads = hidden_size // head_size
        self.length = length
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pooling = pooling

        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            # self.activation = nn.ReLU()
            # self.activation = nn.LeakyReLU()
            self.activation = nn.GELU()
        else:
            self.activation = nn.Softmax(dim=-1)

    def _reshape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_size).transpose(1, 2).contiguous()

    def forward(self, query_states, anchor_states, option=1, mask=None):
        """
        [Opt1]
        query_states: encoder_outputs 
        anchor_states: relevance prompts and encoder_outputs
        [Opt2]
        query_states: relevance prompts (then add encoder_outputs)
        anchor_states: encoder_outputs 
        [Opt3]
        query_states: relevance prompts
        anchor_states: encoder_outputs 
        """
        bsz = query_states.size(0)
        if option==1:
            self.length = 0
            mask = self.expand_mask(mask, anchor_states.size(1)) # L H
            anchor_states = torch.cat([anchor_states, query_states], 1)
        elif option==2:
            self.length = query_states.size(1)
            query_states = torch.cat([query_states, anchor_states], 1)
        elif option==3:
            self.length = -query_states.size(1)

        # base = anchor_states
        Q = self._reshape(self.q_proj(query_states), -1, bsz)
        K = self._reshape(self.k_proj(anchor_states), -1, bsz)
        V = self._reshape(self.v_proj(anchor_states), -1, bsz)

        # aggregate heads
        proj_shape = (bsz * self.num_heads, -1, self.head_size)
        Q = Q.reshape(*proj_shape)
        K = K.reshape(*proj_shape)
        V = V.reshape(*proj_shape)

        # In [opt1] the attention matrix is (bsz*n_heads L (L+N))
        # In [opt2] the attention matrix is (bsz*n_heads (L+N) L)
        attn_logits = torch.bmm(Q, K.transpose(1, 2))   

        # scaling
        scale = math.sqrt(self.head_size)  
        scale = 1
        attn_logits = attn_logits/scale

        # masking
        if mask is not None:
            tgt_len, src_len = attn_logits.size(-2), attn_logits.size(-1)
            attn_logits = attn_logits.view(bsz, self.num_heads, tgt_len, src_len) + mask[:, None, None, :]
            attn_logits = attn_logits.view(bsz*self.num_heads, tgt_len, src_len)

        # probs
        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_output = torch.bmm(attn_probs, V)

        # reshape and revert hidden size
        attn_output = attn_output.view(bsz, self.num_heads, -1, self.head_size)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, -1, self.hidden_size)

        return attn_output

    def expand_mask(self, mask, length=None):
        length = self.length if length is None else length
        if length > 0:
            additional_mask = torch.ones((mask.size(0), self.length), device=mask.device)
            mask = torch.cat([additional_mask, mask], 1)
        elif length < 0:
            mask = torch.ones((mask.size(0), -self.length), device=mask.device)
        return mask

"""
InstanceWisePrompt
    Note that instancewise prompt uses the "prompt" as base.
    And uses the "hidden" as instance-aware features.
"""
class InstanceWisePrompt(nn.Module):
    def __init__(
            self, 
            wte: nn.Embedding, 
            hidden_size: int = 768,
            head_size: int = 64,
            length: int = 1,
            init_idx: Optional[List] = None,
            pooling: str = 'mean',
            activation: str = 'sigmoid',
        ):
        super(InstanceWisePrompt, self).__init__()
        self.orig_embeds = wte
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.length = length
        self.q_proj = nn.Linear(hidden_size, head_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, head_size, bias=True)
        self.pooling = pooling
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()

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

    def forward(self, anchor=None, base=None, input_ids=None, mask=None):
        """
        # reformulatr
        anchor: encoder output token embeddings --> B L H
        base: None --> promt embeddings N H --> B N H

        # adapter
        anchor: relevance embeddings B N H 
        base: encoder output tokens embeddings B L H
        """
        if (anchor is None) and (input_ids is None):
            anchor = self.orig_embeds(input_ids)

        # Reformulator (prompt as base, hidden as anchor)
        if base is None:
            base = self.prompt_embeds
            base = base.repeat(anchor.size(0), 1, 1)

        V = base
        Q = self.q_proj(base) 
        K = self.k_proj(anchor) 
        K = torch.transpose(K, 1, 2)

        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-2)
        # B N/L N
        logits = torch.matmul(Q, K)   

        # if mask is not None: # mask: [B Lx]
        #     expanded_mask = mask[:, :, None].expand(logits.shape)
        #     logits = logits * expanded_mask

        # logits: B N L or B L N --> B N or B L
        # print(logits)
        scale = 1 
        if self.pooling == 'max':
            scores = torch.max(logits / scale, dim=-1).values
        elif self.pooling == 'mean':
            scores = torch.mean(logits / scale, dim=-1)
        elif self.pooling == 'sum':
            scores = torch.sum(logits / scale, dim=-1)

        # scores = self.activation(scores)
        # print(scores.view(4, -1, scores.size(-1)).softmax(-1).topk(10, dim=-1).values[:8, 0])
        # print(scores.view(4, -1, scores.size(-1)).softmax(-1).topk(10, dim=-1).values[:8, 1])

        # B N H * B N or B L H * B L
        return torch.mul(V, scores.unsqueeze(-1))

    def expand_mask(self, mask):
        if self.length > 0:
            additional_mask = torch.ones((mask.size(0), self.length), device=mask.device)
            mask = torch.cat([additional_mask, mask], 1)
        else:
            mask = torch.ones((mask.size(0), -self.length), device=mask.device)
        return mask


class VAE(nn.Module):
    def __init__(
            self,
            hidden_size: int = 768, 
            latent_size: int = 128, 
            learnable_prior: bool = False,
            prefix_length: int = 0,
            **kwargs
        ):
        super(VAE, self).__init__()
        cond_size = kwargs.pop('cond_size', 0)
        self.proj_down = nn.Linear(cond_size+hidden_size, latent_size*2, bias=False)
        self.hidden2mean = nn.Linear(latent_size*2, latent_size, bias=False)
        self.hidden2logv = nn.Linear(latent_size*2, latent_size, bias=False)
        self.proj_up = nn.Linear(cond_size+latent_size, latent_size*2, bias=False)
        self.latent2hidden = nn.Linear(latent_size*2, hidden_size, bias=False)

        if learnable_prior:
            self.proj_down_pri = nn.Linear(hidden_size, latent_size*2, bias=False)
            self.hidden2mean_pri = nn.Linear(latent_size*2, latent_size, bias=False)
            self.hidden2logv_pri = nn.Linear(latent_size*2, latent_size, bias=False)

        # attr
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.prefix_length = prefix_length
        self.kld_kwargs = kwargs.pop('kld_kwargs')

    def forward(self, rel_hidden_state, hidden_state=None, steps=None, conditions=None):
        def _concat_condition(x, c=None):
            if c is not None:
                return torch.cat([c, x], -1)
            else:
                return x
        h_sent_rel = _concat_condition(rel_hidden_state, conditions)

        # [Posterior]
        z_sent_rel = self.proj_down(h_sent_rel)
        mean = self.hidden2mean(z_sent_rel)
        logv = self.hidden2mean(z_sent_rel)
        std = torch.exp(0.5*logv)
        r = torch.rand(mean.shape, device=mean.device) if steps is not None else 0
        z = mean + r * std

        z = _concat_condition(z, conditions)
        z = self.proj_up(z)
        h_sent_rel = self.latent2hidden(z)

        # [Prior]
        if hidden_state is not None:
            h_sent = hidden_state
            z_sent = self.proj_down_pri(h_sent)
            mean_pri = self.hidden2mean_pri(z_sent)
            logv_pri = self.hidden2logv_pri(z_sent)
            loss = kl_loss(logv.view(-1, self.latent_size),
                           mean.view(-1, self.latent_size),
                           logv_pri.view(-1, self.latent_size),
                           mean_pri.view(-1, self.latent_size))
        else:
            loss = kl_loss(logv.view(-1, self.latent_size),
                           mean.view(-1, self.latent_size))
        weight = kl_weight(**self.kld_kwargs, steps=steps)

        return h_sent_rel, loss*weight

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
  