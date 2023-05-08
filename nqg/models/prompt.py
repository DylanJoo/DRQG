"""
TODO: Making this module to be the inherited class of vqg_single_dist
"""
import torch
import inspect
from typing import Optional, Tuple, Union, Dict, Any
from transformers import T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from torch import nn
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from utils import kl_weight, kl_loss, sim_loss
import copy

class SoftEmbedding(nn.Module):
    def __init__(self,
                wte: nn.Embedding,
                n_prompts: int = 1,
                initialize_from_vocab: bool = True, 
                hidden_size: int = 768, 
                latent_size: int = 128):
        super(SoftEmbedding, self).__init__()
        self.n_prompts = n_prompts
        self.orig_embeds = wte
        # initialize with the extra_id_XX tokens
        if initialize_from_vocab:
            self.soft_prompt_embeds = nn.Parameter(
                    self.orig_embeds.weight[-n_prompts:].clone().detach()
            )
        else:
            self.soft_prompt_embeds = nn.Parameter(
                    torch.randn((n_prompts, hidden_size), device=wte.weight.device)
            )
        self.hidden2mean = nn.Linear(hidden_size, latent_size, bias=False)
        self.hidden2logv = nn.Linear(hidden_size, latent_size, bias=False)
        self.latent2hidden = nn.Linear(latent_size, hidden_size, bias=False)
        self.latent_size = latent_size

    def get_soft_prompt_embeds(self):
        return self.soft_prompt_embeds.weights

    def set_gaussian_n_samples_for_generation(self, n_side: int):
        self.std_list = list(range(-n_side, n_side+1, 1))
        self.n_samples = 1 + 2*n_side

    def forward(self, tokens, is_train=False, **kwargs):
        self.loss_KL = 0
        self.loss_COSINE = 0
        batch_size, seq_length = tokens.shape
        e_source = self.orig_embeds(tokens) 
        e_prompt = self.soft_prompt_embeds.unsqueeze(0) # 1, n_prompt, hidden

        # Reparameterize
        if is_train: # variational with gaussian noises
            mean = self.hidden2mean(e_prompt)
            logv = self.hidden2logv(e_prompt)
            std = torch.exp(0.5*logv)
            r = torch.randn(mean.shape, device=e_source.device)
            z = torch.cat([mean, mean+r*std], 0) 
            e_prompt_prime = self.latent2hidden(z) 

            # Concat z to original embeddings
            e_input = torch.cat([
                e_prompt_prime.repeat(batch_size//2, 1, 1), e_source 
            ], 1)

            # compute loss
            loss = kl_loss(logv.view(-1, self.latent_size),
                           mean.view(-1, self.latent_size))
            weight = kl_weight(**kwargs)
            self.loss_KL = loss * weight
            self.loss_COSINE = sim_loss(mean, mean+r*std)

        # evaluation  # batch would be the same as number of passage 
        else: 
            mean = self.hidden2mean(e_prompt)
            logv = self.hidden2logv(e_prompt)
            std = torch.exp(0.5*logv)
            z = torch.cat([mean+std*i for i in self.std_list], 0)
            e_prompt_prime = self.latent2hidden(z)

            # Concat z to original embeddings
            # e_prompt: n_samples, n_prompt, hidden --> (n_samples*bs), n_prompt, hidden
            # e_source: bs, n_prompt, hidden --> (bs[0]*n_samples...), n_prompt, hidden
            e_input = torch.cat([
                e_prompt_prime.repeat(batch_size, 1, 1), 
                torch.repeat_interleave(e_source, self.n_samples, dim=0)
            ], 1)

        return e_input
