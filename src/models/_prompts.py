import math
import torch
from typing import Optional, Tuple, Union, Dict, Any, List
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.bart.modeling_bart import BartAttention
from utils import kl_weight, kl_loss
from models.utils import get_activation

class SoftPrompt(nn.Module):
    def __init__(
            self, 
            wte: nn.Embedding,
            hidden_size: int = 768,
            head_size: int = 64,
            init_ids: Optional[List] = [],
            activation: Optional[str] = 'sigmoid',
        ):
        super(AdaPrompt, self).__init__()
        # config from `_base`
        self.orig_embeds = wte
        self.hidden_size = hidden_size

        # config for `SoftPrompt`
        self.init_idx = init_idx # will set them later
        self.head_size = head_size
        self.length = len(init_ids) # [BUG] change to longer 
        self.activation = get_activation(activation)

        # NN modules for SoftPrompt
        self.prompts = nn.Parameter(
                torch.randn((len(init_ids), hidden_size), 
                    device=wte.weight.device)
        )
        self.rel_prompts = nn.Parameter(
                torch.randn((len(init_rel_ids), hidden_size), 
                    device=wte.weight.device)
        )
        self.q_proj = nn.Linear(hidden_size, head_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, head_size, bias=True)

    def forward(self, input_ids, relevance, steps=None):
        input_embeds = self.orig_embeds(input_ids)
        device = input_embeds.device

        # self.n_samples = len(relevance) // hidden_states_src.size(0)
        doc_prompts = self.reformulator(anchor=hidden_states_src, base=self.prompts)
        ##
        # Reformulator (prompt as base, hidden as anchor)
        prompts = self.prompts.repeat(input_ids.size(0), 1, 1)
        if base is None:
            base = self.prompt_embeds)
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
        scores = torch.mean(logits / scale, dim=-1)

        # scores = self.activation(scores)
        # print(scores.view(4, -1, scores.size(-1)).softmax(-1).topk(10, dim=-1).values[:8, 0])
        # print(scores.view(4, -1, scores.size(-1)).softmax(-1).topk(10, dim=-1).values[:8, 1])

        # B N H * B N or B L H * B L
        return torch.mul(V, scores.unsqueeze(-1))
        ##
        # Relevance wise prompt
        relevance = torch.cat([(1-relevance).view(-1, 1), relevance.view(-1, 1)], 1)
        relevance = relevance.to(device)
        hidden_states_rel = torch.matmul(relevance, self.label_prompts).unsqueeze(1) 
        rel_prompts = self.reformulator(anchor=hidden_states_rel, base=self.prompts)

        # [TODO] Maybe try reformulate the prompts
        self.length = 2*len(self.init_idx) # output should fit the mask
        return torch.cat([doc_prompts, rel_prompts, hidden_states_src], 1)

    def expand_mask(self, mask):
        mask = mask.repeat(self.n_samples, 1)
        additional_mask = torch.ones(
                (mask.size(0), self.length), device=mask.device
        )
        mask = torch.cat([additional_mask, mask], -1)
        return mask

    def set_embeddings(self):
        self.prompts = nn.Parameter(
                self.orig_embeds.weight[self.init_idx].clone().detach()
        )
        self.label_prompts = nn.Parameter(
                self.orig_embeds.weight[self.lbl_init_idx].clone().detach()
        )
        print(f"{self.__class__.__name__} embeddings set: ", self.init_idx, self.lbl_init_idx)

class MultiheadAttentive(nn.Module):
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
        scores = torch.mean(logits / scale, dim=-1)

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
