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

