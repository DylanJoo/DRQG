import torch
import inspect
from typing import Optional, Tuple, Union, Dict, Any, List
from torch import nn
from utils import kl_weight, kl_loss, PairwiseCELoss
import copy

class SoftEmbedding(nn.Module):
    def get_pairwise_loss(self):
        return self.pairwise_loss

    def get_KL_loss(self):
        return self.kld_loss * self.kld_weight

    def set_gaussian_range(self, std_list):
        self.std_list = std_list

    def post_init(self):
        pass

    def __init__(
            self,
            wte: nn.Embedding,
            hidden_size: int = 768, 
            latent_size: int = 128, 
            n_prompts: int = 1,
            initialize_from_vocab: bool = False, 
            used_prompt_idx: List[int] = None, 
            pooler: Optional[object] = None,
            adaptive_pooling: Optional[str] = None,
            **kwargs
        ):
        super(SoftEmbedding, self).__init__()
        # [NOTE] If prepending <s> is needed
        self.n_prompts = n_prompts
        self.orig_embeds = wte

        if initialize_from_vocab:
            if isinstance(used_prompt_idx, list):
                used_prompt_idx = used_prompt_idx*self.n_prompts
                used_prompt_idx = used_prompt_idx[:self.n_prompts]
            else:
                vocab_size = wte.weight.size(-1) - 1
                used_prompt_idx = list(
                        range(vocab_size-self.n_prompts, vocab_size, 1)
                )
            self.prompt_embeds = nn.Parameter(
                    self.orig_embeds.weight[used_prompt_idx].clone().detach()
            )
        else:
            self.prompt_embeds = nn.Parameter(
                    torch.randn((self.n_prompts, hidden_size), device=wte.weight.device)
            )

        if kwargs.pop("has_compressed_layer"):
            self.hidden2mean = nn.Linear(2*latent_size, latent_size, bias=False)
            self.hidden2logv = nn.Linear(2*latent_size, latent_size, bias=False)
            self.latent2hidden = nn.Linear(latent_size*2, hidden_size, bias=False)

            self.downproject = nn.Linear(hidden_size, latent_size*2, bias=False)
            self.upproject = nn.Linear(latent_size, latent_size*2, bias=False)
        else:
            self.hidden2mean = nn.Linear(hidden_size, latent_size, bias=False)
            self.hidden2logv = nn.Linear(hidden_size, latent_size, bias=False)
            self.latent2hidden = nn.Linear(latent_size, hidden_size, bias=False)
            self.upproject = None
            self.downproject = None

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.pooler = pooler
        self.adaptive_pooling = adaptive_pooling
        self.kld_kwargs = kwargs
        self.post_init()

    def forward(self, tokens, is_train=False, steps=1, clf_labels=None):
        """ `training` and `evaluation/prediction` phrases have diff setups. 
        The training batch contains both (specified) batch_size * 2 
        as exact mini-batch during training; while evaluation batch is unmodified

        e_source: the original token embeddings.
        e_prompt: the prompts embeddings.

        Reparameterization 
            Convert e_prompt from (dynamic) hidden space into continuous space Z 
            Convert continuous embedding in Z to hidden space
        
        Concat 
            e (variational prompt) and e_source
        """

        batch_size, seq_length = tokens.shape
        e_source = self.orig_embeds(tokens)
        e_prompt = self.prompt_embeds.unsqueeze(0) 
        if self.downproject is not None:
            e_prompt = self.downproject(e_prompt)

        if is_train:
            # (1, Np, H)-->(1, Np, Hz)-->(1*2, Np, H)
            mean = self.hidden2mean(e_prompt)
            logv = self.hidden2logv(e_prompt)
            std = torch.exp(0.5*logv)
            r = torch.randn(mean.shape, device=e_source.device)
            z = torch.cat([
                mean+torch.randn(mean.shape, device=mean.device)*(1-i)*std \
                    for i in clf_labels
            ], 0)
            # z = mean + r * std
            if self.upproject is not None:
                z = self.upproject(z)
            e = self.latent2hidden(z) 

            # [reshape] All done

            # compute loss
            loss = kl_loss(logv.view(-1, self.latent_size),
                           mean.view(-1, self.latent_size))
            self.kld_loss = loss
            self.kld_weight = kl_weight(**self.kld_kwargs, steps=steps)

        else: 
            mean = self.hidden2mean(e_prompt)
            logv = self.hidden2logv(e_prompt)
            std = torch.exp(0.5*logv)
            z = torch.cat([mean+std*i for i in self.std_list], 0)
            if self.upproject is not None:
                z = self.upproject(z)
            e = self.latent2hidden(z)

            # [reshape]
            e = e.repeat_interleave(batch_size, dim=0)
            e_source = e_source.repeat(len(self.std_list), 1, 1)

            self.kld_loss = 0
            self.kld_weight = 0

        # [concat]
        e_input = torch.cat([e, e_source], 1)

        return e_input

class SoftAdaptiveEmbedding2(SoftEmbedding):
    """ 
    [TODO] revise this doc
    Preprocess
        e_pooled is the average sentence embeddings

    Reparameterization 
        Convert e_pooled from hidden space into continuous space Z 
        Convert continuous embedding in Z to hidden space
    
    Prompt process
        Summation of (variational sentence embeddings) and (e_prompt)

    Concat 
        e (prompt with the mean of variational source) and e_source
    """

    def post_init(self, **kwargs):
        self.downproject = nn.Linear(
                self.hidden_size+2, self.latent_size*2,
                bias=False
        )
        self.upproject = nn.Linear(
                self.latent_size+2, self.latent_size*2, 
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
            e_pooled = torch.cat([e_pooled, cond], -1)
            if self.downproject is not None:
                e_pooled = self.downproject(e_pooled)

            mean = self.hidden2mean(e_pooled)
            logv = self.hidden2logv(e_pooled)
            std = torch.exp(0.5*logv)
            r = torch.randn(mean.shape, device=e_source.device)
            z = mean + r * std 

            z = torch.cat([z, cond], -1)
            if self.upproject is not None:
                z = self.upproject(z)
            e = self.latent2hidden(z) 

            # [reshape] All done
            # (1, N, H) --> (B, N, H)
            e_prompt = e_prompt.repeat(batch_size, 1, 1)
            # (B, N, H) + (B, 1, H) = (B, N, H)
            e = e_prompt + e 

            # compute loss
            loss = kl_loss(logv.view(-1, self.latent_size),
                           mean.view(-1, self.latent_size))
            self.kld_loss = loss / len(clf_labels)
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
            e_pooled = torch.cat([e_pooled, cond], -1)
            if self.downproject is not None:
                e_pooled = self.downproject(e_pooled)

            mean = self.hidden2mean(e_pooled)
            z = mean 

            z = torch.cat([z, cond], -1)
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

    # def forward(self, tokens, is_train=False, steps=1, clf_labels=None):
    #     batch_size, seq_length = tokens.shape
    #     e_source = self.orig_embeds(tokens)
    #     e_prompt = self.prompt_embeds.unsqueeze(0) 
    #
    #     if is_train:
    #         # simple onehot
    #         cond = torch.cat([(1-clf_labels).view(-1, 1), clf_labels.view(-1, 1)], 1)
    #         e_cond = (cond @ self.condition_embeds.to(cond.device)).unsqueeze(1)
    #         # (1, Np, H)-->(1, Np, Hz)-->(1*2, Np, H)
    #         e_prompt = e_prompt.repeat(batch_size, 1, 1)
    #         e_prompt = torch.cat([e_prompt, e_cond], 1)
    #         if self.downproject is not None:
    #             e_prompt = self.downproject(e_prompt)
    #
    #         mean = self.hidden2mean(e_prompt)
    #         logv = self.hidden2logv(e_prompt)
    #         std = torch.exp(0.5*logv)
    #         r = torch.randn(mean.shape, device=e_source.device)
    #         z = mean + r * std
    #         if self.upproject is not None:
    #             z = self.upproject(z)
    #         z = torch.cat([z, e_cond], 1)
    #         e = self.latent2hidden(z) 
    #
    #         # [reshape] All done
    #
    #         # compute loss
    #         loss = kl_loss(logv.view(-1, self.latent_size),
    #                        mean.view(-1, self.latent_size))
    #         self.kld_loss = loss
    #         self.kld_weight = kl_weight(**self.kld_kwargs, steps=steps)
    #
    #     else: 
    #         clf_labels = torch.range(0, len(self.std_list)-1, 1)/(len(self.std_list)-1)
    #         cond = torch.cat([(1-clf_labels).view(-1, 1), clf_labels.view(-1, 1)], 1)
    #         e_cond = (cond @ self.condition_embeds.to(cond.device)).unsqueeze(1)
    #         if self.downproject is not None:
    #             e_prompt = self.downproject(e_prompt)
    #
    #         mean = self.hidden2mean(e_prompt)
    #         logv = self.hidden2logv(e_prompt)
    #         std = torch.exp(0.5*logv)
    #         z = torch.cat([
    #             torch.cat([mean+std, e_cond], 1) for i in self.std_list
    #         ], 0)
    #         if self.upproject is not None:
    #             z = self.upproject(z)
    #         z = torch.cat([z, e_cond], 1)
    #         e = self.latent2hidden(z)
    #
    #         # [reshape]
    #         e = e.repeat_interleave(batch_size, dim=0)
    #         e_source = e_source.repeat(len(self.std_list), 1, 1)
    #
    #         self.kld_loss = 0
    #         self.kld_weight = 0
    #
    #     # [concat]
    #     e_input = torch.cat([e, e_source], 1)
    #
    #     return e_input

class SoftAdaptiveEmbedding(SoftEmbedding):
    """ 
    Preprocess
        e_pooled is the average sentence embeddings

    Reparameterization 
        Convert e_pooled from hidden space into continuous space Z 
        Convert continuous embedding in Z to hidden space
    
    Prompt process
        Summation of (variational sentence embeddings) and (e_prompt)

    Concat 
        e (prompt with the mean of variational source) and e_source
    """

    def forward(self, tokens, is_train=False, steps=1, clf_labels=None):
        batch_size, seq_length = tokens.shape
        e_source = self.orig_embeds(tokens)
        e_prompt = self.prompt_embeds.unsqueeze(0) 

        # [pooling] mean or max
        if self.adaptive_pooling == 'mean':
            e_pooled = torch.mean(e_source, dim=1).unsqueeze(1)
        elif self.adaptive_pooling == 'max':
            e_pooled = torch.max(e_source, dim=1).values.unsqueeze(1)

        if self.downproject is not None:
            e_pooled = self.downproject(e_pooled)

        if is_train: 
            mean = self.hidden2mean(e_pooled)
            logv = self.hidden2logv(e_pooled)
            std = torch.exp(0.5*logv)
            r = torch.randn(mean.shape, device=e_source.device)
            z  = mean + r * std 
            if self.upproject is not None:
                z = self.upproject(z)
            e = self.latent2hidden(z) 

            # [reshape] All done
            # (1, N, H) --> (B, N, H)
            e_prompt = e_prompt.repeat(batch_size, 1, 1)
            # (B, N, H) + (B, 1, H) = (B, N, H)
            scales = clf_labels.view(-1, 1, 1).repeat(1, e.size(1), e.size(2))
            e = e_prompt + e * (1-scales)

            # compute loss
            loss = kl_loss(logv.view(-1, self.latent_size),
                           mean.view(-1, self.latent_size))
            self.kld_loss = loss / len(clf_labels)
            self.kld_weight = kl_weight(**self.kld_kwargs, steps=steps)
        else: 
            mean = self.hidden2mean(e_pooled)
            logv = self.hidden2logv(e_pooled)
            std = torch.exp(0.5*logv)
            z = torch.cat([mean+i*std for i in self.std_list], 0)
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

class SoftEmbeddingWithPooler(SoftEmbedding):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.pooler is not None, \
                'the pooler was not succesfully assigned.'

    def forward(self, tokens, is_train=False, steps=1, clf_labels=None):
        seq_length = tokens.size()[-1]
        e_input = super().forward(tokens, is_train, steps, clf_labels)
        pooled_output = self.pooler(e_input, None, None)[0]

        return pooled_output

class SoftStaticEmbedding(SoftEmbedding):
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

class SoftEmbeddingForDecoder(SoftEmbedding):

    def __init__(
            self,
            wte: nn.Embedding,
            hidden_size: int = 768, 
            latent_size: int = 128, 
            n_prompts: int = 1,
            initialize_from_vocab: bool = False, 
            pooler: Optional[object] = None,
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

# class SoftResidualEmbedding(SoftEmbedding):
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.prompt_embeds_base = nn.Parameter(
#                 torch.randn(
#                     (self.n_prompts, self.hidden_size), 
#                     device=self.orig_embeds.weight.device
#                 )
#         )
#     def forward(self, tokens, is_train=False, steps=1, clf_labels=None):
#         batch_size, seq_length = tokens.shape
#         e_source = self.orig_embeds(tokens)
#         e_prompt = self.prompt_embeds.unsqueeze(0) 
#         if self.downproject is not None:
#             e_prompt = self.downproject(e_prompt)
#
#         if is_train: 
#             mean = self.hidden2mean(e_prompt)
#             logv = self.hidden2logv(e_prompt)
#             std = torch.exp(0.5*logv)
#             z = torch.cat([
#                 mean + torch.randn(mean.shape, device=e_source.device)*(1-i)*std \
#                         for i in clf_labels
#             ], 0) 
#             if self.upproject is not None:
#                 z = self.upproject(z)
#             e = self.latent2hidden(z) 
#
#             if self.prompt_embeds_base is not None:
#                 e = e + self.prompt_embeds_base.unsqueeze(0) 
#             
#             # [reshape] All done
#
#             # compute loss
#             loss = kl_loss(logv[clf_labels==1].view(-1, self.latent_size),
#                            mean[clf_labels==1].view(-1, self.latent_size))
#             self.kld_loss = loss / len(clf_labels==1)
#             self.kld_weight = kl_weight(**self.kld_kwargs, steps=steps)
#
#         else: 
#             mean = self.hidden2mean(e_prompt)
#             logv = self.hidden2logv(e_prompt)
#             std = torch.exp(0.5*logv)
#             z = torch.cat([mean+i*std for i in self.std_list], 0)
#             if self.upproject is not None:
#                 z = self.upproject(z)
#             e = self.latent2hidden(z) 
#
#             if self.prompt_embeds_base is not None:
#                 e = e + self.prompt_embeds_base.unsqueeze(0) 
#
#             # [reshape]
#             e = e.repeat_interleave(batch_size, dim=0)
#             e_source = e_source.repeat(len(self.std_list), 1, 1)
#
#             # compute loss
#             self.kld_loss = 0
#             self.kld_weight = 0
#
#         # Concat z to original embeddings
#         e_input = torch.cat([e, e_source], 1)
#
#         return e_input
