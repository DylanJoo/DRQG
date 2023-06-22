import math
import torch
from typing import Optional, Tuple, Union, Dict, Any, List
from torch import nn
from torch.nn import functional as F
from utils import kl_weight, kl_loss
import copy
from .modules import InstanceWisePrompt
from torch.nn import CrossEntropyLoss

""" 
DocRelPrompt
    Document-relevance-awared prompt for confitionalQG.
    Used components include 'document-wise' and 'relevant-wise' prompts.
    The prompts are treated as the base, while `doc and rel` are anchors.

    [NOTE] the token-level cross-attention seems failed to perform controlled generation.
    [TODO] 
"""
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
                activation='tanh'
        )
        self.label_reformulator = InstanceWisePrompt(
                wte=None,
                hidden_size=hidden_size,
                head_size=head_size,
                activation='tanh'
        )

    def forward(self, relevance, hidden_states_src=None, input_ids=None, steps=None):
        if hidden_states_src is None:
            hidden_states_src = self.orig_embeds(input_ids)
        self.n_samples = len(relevance) // hidden_states_src.size(0)

        # Document-wise prompt
        doc_prompts = self.reformulator(
                anchor=hidden_states_src, base=self.prompts
        )

        # Relevance wise prompt
        relevance = torch.cat([(1-relevance).view(-1, 1), relevance.view(-1, 1)], 1)
        relevance = relevance.to(self.label_prompts.device)
        hidden_states_rel = torch.matmul(relevance, self.label_prompts).unsqueeze(1) 
        rel_prompts = self.reformulator(
                anchor=hidden_states_rel, base=self.prompts
        )

        # [TODO] Maybe try reformulate the prompts
        self.length = 2*len(self.init_idx) # output should fit the mask
        return torch.cat([doc_prompts, rel_prompts, hidden_states_src], 1)

    def calculate_src_ibn_loss(self, hidden_state, bs):
        if hidden_state.size(1) != 1:
            hidden_state = hidden_state.mean(1)[:, None, :]
        hidden_size = hidden_state.size(-1)
        doc_hidden_state = hidden_state.view(bs, -1, hidden_size)

        doc_scores = torch.bmm(
                doc_hidden_state, 
                doc_hidden_state.transpose(-1, -2)
        )
        loss_fct = CrossEntropyLoss()
        n_size = doc_scores.size(1)
        doc_labels = torch.arange(0, n_size, 
                device=hidden_state.device
        )
        docibn_loss = loss_fct(
                doc_scores.view(-1, n_size), doc_labels.repeat(bs)
        )
        return docibn_loss

    def expand_mask(self, mask):
        mask = mask.repeat(self.n_samples, 1)
        additional_mask = torch.ones((mask.size(0), self.length), device=mask.device)
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
