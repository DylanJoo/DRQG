import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5Stack
from utils import kl_loss
from torch.nn import CosineEmbeddingLoss

class VariationalWrapper(nn.Module):
    def __init__(self, decoder, vae_config, use_checkpoint=False):
        super().__init__()

        # reparemeterize settings
        self.vae_config = vae_config
        self.latent_size = self.vae_config.latent_size
        self.hidden2pmean = nn.Linear(decoder.config.d_model, vae_config.latent_size)
        self.hidden2nmean = nn.Linear(decoder.config.d_model, vae_config.latent_size)
        self.hidden2plogv = nn.Linear(decoder.config.d_model, vae_config.latent_size)
        self.hidden2nlogv = nn.Linear(decoder.config.d_model, vae_config.latent_size)
        self.latent2hidden = nn.Linear(vae_config.latent_size, decoder.config.d_model)

        # original t5
        self.decoder = decoder

        # checkpoint wrapper
        ## [IMPORTANT] make sure this decoder wrapper can be checkpointed.
        block = []
        for mod in self.decoder:
            wrapped_mod = CheckpointWrapper(mod)


    def forward(self, encoder_hidden_states=None, **kwargs):
        """
        Wrap the variational layer to replace original decoder.

        encoder_hidden_states: `tensor`
            the hidden states of encoder output on each T5 layer.
        **kwargs:
            the other input argument for T5 decoder
        """
        batch_size, seq_length, d_model = encoder_hidden_states.shape
        pn_boundary = batch_size // 2

        # [NOTE] Transform it into a single vector (i.e., d dimensions x 1 tokens) with mask
        # [NOTE] Thinking of adopting a single random vector to align two distribution.
        r = torch.randn([pn_boundary, 1, self.latent_size]).to(encoder_hidden_states.device)
        pmean = self.hidden2pmean(encoder_hidden_states[:pn_boundary, :1, :])
        plogv = self.hidden2plogv(encoder_hidden_states[:pn_boundary, :1, :])
        pstd = torch.exp(0.5 * plogv)
        z = r * pstd + pmean
        positive = self.latent2hidden(z)

        nmean = self.hidden2nmean(encoder_hidden_states[pn_boundary:, :1, :])
        nlogv = self.hidden2nlogv(encoder_hidden_states[pn_boundary:, :1, :])
        nstd = torch.exp(0.5 * nlogv)
        z = r * nstd + nmean
        negative = self.latent2hidden(z)
        zeros = torch.zeros(batch_size, seq_length-1, d_model).to(encoder_hidden_states.device)

        residuals = torch.cat((torch.cat((positive, negative), 0), zeros), 1)
        encoder_hidden_states = encoder_hidden_states + residuals

        # calculate loss
        self.compute_loss_reparam(
                pmean=pmean, nmean=nmean, plogv=plogv, nlogv=nlogv
        )
        output = self.decoder(
                encoder_hidden_states=encoder_hidden_states, **kwargs
        )
        return output
    
    def compute_loss_reparam(self, pmean, nmean, plogv, nlogv):
        # kl of positive dist.
        loss_kl_pos = kl_loss(
            plogv.view(-1, self.latent_size), 
            pmean.view(-1, self.latent_size)
        )
        # kl of negative dist.
        loss_kl_neg = kl_loss(
            nlogv.view(-1, self.latent_size), 
            nmean.view(-1, self.latent_size)
        )
        # cosine of two dists. (the expected values are zero)
        # [NOTE] It may be another value or margins.
        cosine_loss = CosineEmbeddingLoss()
        zeros = torch.tensor([-1] * pmean.shape[0]).to(pmean.device)
        loss_cosine = cosine_loss(
            pmean.view(-1, self.latent_size),
            nmean.view(-1, self.latent_size),
            zeros
        )
        self.loss_reparam = loss_kl_pos + loss_kl_neg + loss_cosine
        # loss_kl_w = kl_weight(
        #     self.vae_config.annealing_fn, 
        #     steps, 
        #     self.vae_config.k, 
        #     self.vae_config.x0
        # )
