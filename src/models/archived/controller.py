import math
import torch
from typing import Optional, Tuple, Union, Dict, Any, List
from torch import nn
from torch.nn import functional as F
from utils import kl_weight, kl_loss
import copy
from .modules import InstanceWisePrompt, AttentivePrompt
from torch.nn import CrossEntropyLoss
from transformers.models.bart.modeling_bart import BartAttention

""" 
DocRelPrompt
    Document-relevance-awared prompt for confitionalQG.
    Used components include 'document-wise' and 'relevant-wise' prompts.
    The prompts are treated as the base, while `doc and rel` are anchors.
[TODO] Add VAE components
"""

class DocRelPrompt(nn.Module):
    def __init__(
            self, 
            wte: nn.Embedding,
            hidden_size: int = 768,
            head_size: int = 64,
            lbl_init_idx: Optional[List] = None,
            init_idx: Optional[List] = None,
            activation: Optional[str] = 'sigmoid',
        ):
        super(DocRelPrompt, self).__init__()
        self.orig_embeds = wte
        self.lbl_init_idx = lbl_init_idx
        self.init_idx = init_idx

        self.prompts = nn.Parameter(torch.randn(
            (len(init_idx), hidden_size), device=wte.weight.device
        ))
        self.label_prompts = nn.Parameter(torch.randn(
            (len(lbl_init_idx), hidden_size), device=wte.weight.device
        ))
        self.reformulator = InstanceWisePrompt(
                wte=None,
                hidden_size=hidden_size,
                head_size=head_size,
                activation=activation
        )
        self.label_reformulator = InstanceWisePrompt(
                wte=None,
                hidden_size=hidden_size,
                head_size=head_size,
                activation=activation
        )

    def forward(self, relevance, hidden_states_src=None, input_ids=None, steps=None):
        if hidden_states_src is None:
            hidden_states_src = self.orig_embeds(input_ids)

        device = hidden_states_src.device
        self.n_samples = len(relevance) // hidden_states_src.size(0)

        # Document-wise prompt
        doc_prompts = self.reformulator(anchor=hidden_states_src, base=self.prompts)

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

""" 
RelAdapter
    Relevance adapter for relevantQG 
    Used components include 'relevance-awared' adapters.

[TODO] Add VAE components
[TODO] Try different relevance modeling for relevant-wise prompts.
"""
class RelAdapter(nn.Module):
    def __init__(
            self, 
            wte: nn.Embedding,
            hidden_size: int = 768,
            head_size: int = 64,
            pos_init_idx: Optional[List] = None,
            neg_init_idx: Optional[List] = None,
            pooling: str = 'mean',
            activation: str = 'tanh'
        ):
        super(RelAdapter, self).__init__()
        self.orig_embeds = wte
        self.pos_init_idx = pos_init_idx
        self.neg_init_idx = neg_init_idx
        self.hidden_size = hidden_size
        self.device = wte.weight.device

        # [TODO] Revise other prompt settings
        self.pos_prompts = nn.Embedding(len(pos_init_idx), hidden_size)
        self.neg_prompts = nn.Embedding(len(neg_init_idx), hidden_size)

        self.adapter = AttentivePrompt(
                wte=None,
                hidden_size=hidden_size,
                head_size=head_size,
                pooling=pooling,
                activation=activation
        )

    def forward(self, relevance, hidden_states_src=None, mask=None, use_residual=False):
        device = hidden_states_src.device
        if relevance is None:
            relevance = torch.ones(hidden_states_src.size(0), device=device)

        batch_doc_size = hidden_states_src.size(0)
        # [relevance conditioned]
        relevance = relevance.to(device)
        pos_prompts = self.pos_prompts.weight.unsqueeze(0).repeat(batch_doc_size, 1, 1)
        neg_prompts = self.neg_prompts.weight.unsqueeze(0).repeat(batch_doc_size, 1, 1)
        rel_prompts = (relevance.view(-1, 1, 1) * pos_prompts + \
                       (1-relevance).view(-1, 1, 1) * neg_prompts) 

        # [condition settings]
        # residual = self.adapter(hidden_states_src, rel_prompts)
        residual = self.adapter(rel_prompts, hidden_states_src, option=3, mask=mask)

        # B L 
        if use_residual:
            return residual + hidden_states_src
        else:
            return residual

    def set_embeddings(self):
        self.pos_prompts = self.pos_prompts.from_pretrained(
                self.orig_embeds.weight[self.pos_init_idx].clone().detach()
        )
        self.neg_prompts = self.neg_prompts.from_pretrained(
                self.orig_embeds.weight[self.neg_init_idx].clone().detach()
        )
        print(f"{self.__class__.__name__} embeddings set: ", \
                self.pos_init_idx, self.neg_init_idx)
