import torch
import inspect
from typing import Optional, Tuple, Union, Dict, Any, List
from torch import nn
from utils import kl_weight, kl_loss
import copy

class SoftEmbedding(nn.Module):
    """
    SoftEmbeddings as additional concatenated tokens. 
    (i.e., Prompt-learning)
    """
    def get_KL_loss(self):
        if self.kld_loss:
            return self.kld_loss * self.kld_weight
        else:
            return 0
    def set_gaussian_range(self, std_list):
        self.std_list = std_list
    def post_init(self):
        return 0

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
        super(SoftEmbedding, self).__init__()
        # [NOTE] If prepending <s> is needed
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

    def forward(self, tokens, is_train=False, steps=1, clf_labels=None):
        batch_size, seq_length = tokens.shape
        e_source = self.orig_embeds(tokens)
        e_prompt = self.prompt_embeds.unsqueeze(0) 
        self.kld_loss = None

        if is_train:
            e_prompt = e_prompt.repeat(batch_size, 1, 1)
            e_input = torch.cat([e_prompt, e_source], 1)
        else: 
            # [NOTE] to make all the soft embed run, replicate the sample without variation
            sample_size = len(self.std_list)
            e_prompt = e_prompt.repeat(sample_size*batch_size, 1, 1)
            e_source = e_source.repeat(len(self.std_list), 1, 1)
            e_input = torch.cat([e_prompt, e_source], 1)

        return e_input

class SoftBasicEmbedding(SoftEmbedding):
    """
    Soft controllable prompt
    (i.e., Controllable-VAE prompt)
    """

    def post_init(self):
        if self.has_compressed_layer:
            self.hidden2mean = nn.Linear(2*self.latent_size, self.latent_size, bias=False)
            self.hidden2logv = nn.Linear(2*self.latent_size, self.latent_size, bias=False)
            self.latent2hidden = nn.Linear(self.latent_size*2, self.hidden_size, bias=False)

            self.downproject = nn.Linear(self.hidden_size, self.latent_size*2, bias=False)
            self.upproject = nn.Linear(self.latent_size, self.latent_size*2, bias=False)
        else:
            self.hidden2mean = nn.Linear(self.hidden_size, self.latent_size, bias=False)
            self.hidden2logv = nn.Linear(self.hidden_size, self.latent_size, bias=False)
            self.latent2hidden = nn.Linear(self.latent_size, self.hidden_size, bias=False)

            self.downproject = None
            self.upproject = None

    def forward(self, tokens, is_train=False, steps=1, clf_labels=None):
        """ 
        Reparameterization 
            Convert e_prompt from (dynamic) hidden space into continuous space Z 
            Convert continuous embedding in Z to hidden space
        
        Concat 
            e (variational prompt) and e_source
        """
        batch_size, seq_length = tokens.shape
        e_source = self.orig_embeds(tokens)
        e_prompt = self.prompt_embeds.unsqueeze(0) 

        if is_train:
            e_prompt = e_prompt.repeat(batch_size, 1, 1)
            cond = torch.cat([(1-clf_labels).view(-1, 1), clf_labels.view(-1, 1)], 1)
            cond = cond.view(-1, 1, 2)
            cond = cond.repeat(1, e_prompt.size(1), 1)
            e_prompt = torch.cat([e_prompt, cond], -1)
            if self.downproject is not None:
                e_prompt = self.downproject(e_prompt)

            mean = self.hidden2mean(e_prompt)
            logv = self.hidden2logv(e_prompt)
            std = torch.exp(0.5*logv)
            r = torch.randn(mean.shape, device=e_source.device)
            z = mean + r * std

            z = torch.cat([z, cond], -1)
            if self.upproject is not None:
                z = self.upproject(z)
            e = self.latent2hidden(z) 

            # compute loss
            loss = kl_loss(logv.view(-1, self.latent_size),
                           mean.view(-1, self.latent_size))
            self.kld_loss = loss / batch_size
            self.kld_weight = kl_weight(**self.kld_kwargs, steps=steps)
        else: 
            n_samples = len(self.std_list)
            e_prompt = e_prompt.repeat(batch_size*n_samples, 1, 1)
            clf_labels = torch.range(0, n_samples-1, 1)/(n_samples-1)
            cond = torch.cat([(1-clf_labels).view(-1, 1), clf_labels.view(-1, 1)], 1)
            cond = cond.view(-1, 1, 2).to(e_prompt.device)
            cond = cond.repeat(1, e_prompt.size(1), 1)
            cond = cond.repeat_interleave(batch_size, dim=0)
            e_prompt = torch.cat([e_prompt, cond], -1)
            if self.downproject is not None:
                e_prompt = self.downproject(e_prompt)

            mean = self.hidden2mean(e_prompt)

            z = torch.cat([mean, cond], -1)
            if self.upproject is not None:
                z = self.upproject(z)
            e = self.latent2hidden(z)

            # [reshape]
            e_source = e_source.repeat(len(self.std_list), 1, 1)
            self.kld_loss = None

        # [concat]
        e_input = torch.cat([e, e_source], 1)

        return e_input

class SoftAdaptiveEmbedding(SoftBasicEmbedding):
    """
    Document-relevance-awared via soft-prompt 
    (i.e., Controllable-VAE document representation using soft-prompt)
    """
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

class SoftStaticEmbedding(SoftBasicEmbedding):
    """ 
    Reparameterization 
        Convert e_source from hidden space into continuous space Z 
        Convert continuous embedding in Z to hidden space
    
    Prompt process
        Summation of (average e among length dimension) and (e_prompt)

    Concat 
        e (prompt with the mean of variational source) and e_source
    """
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
            z = mean + r * std
            if self.upproject is not None:
                z = self.upproject(z)
            e = self.latent2hidden(z) 

            # [reshape] 
            # (1, N, H) --> (B, N, H)
            e_prompt = e_prompt.repeat(batch_size, 1, 1)
            # (B, N, H) + (B, 1, H) = (B, N, H)
            e = e_prompt + torch.mean(e, dim=1).unsqueeze(1)

            # compute loss
            loss = kl_loss(logv[clf_labels==1].view(-1, self.latent_size),
                           mean[clf_labels==1].view(-1, self.latent_size))
            self.kld_loss = loss / len(clf_labels==1)
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

class SoftEmbeddingForDecoder(SoftBasicEmbedding):

    def __init__(
            self,
            wte: nn.Embedding,
            hidden_size: int = 768, 
            latent_size: int = 128, 
            n_prompts: int = 1,
            initialize_from_vocab: bool = False, 
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

