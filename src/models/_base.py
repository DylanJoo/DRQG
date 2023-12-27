import torch
from torch import nn
from transformers import T5ForConditionalGeneration

class FlanT5(T5ForConditionalGeneration):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.n_samples = 1

    def forward(self, 
                steps=None, 
                rel_labels=None, 
                rel_scores=None, 
                passage=None, 
                **kwargs):
        return super().forward(**kwargs)


class VAE(nn.Module):

    def __init__(self, config, latent_size):
        super().__init__()
        self.d_model = config.d_model
        self.latent_size = lsz = latent_size
        self.hidden2mean = nn.Linear(config.d_model, lsz)
        self.hidden2mean = nn.Linear(config.d_model, lsz)
        self.hidden2logv = nn.Linear(config.d_model, lsz)

        self.num_decoder_layers = config.num_decoder_layers
        self.num_heads = config.num_heads

        # the memory mechanism
        msz = config.num_decoder_layers * config.d_model
        self.decoder = nn.Linear(lsz, msz, bias=False)

    def _reshape(self, hidden_state_r):
        self_attn = hidden_state_r.reshape(
                self.num_decoder_layers,
                hidden_state_r.shape[0],
                self.num_heads,
                1,
                self.d_model // self.num_heads
        )
        past_key_values = tuple((sa, sa) for sa in self_attn)
        return past_key_values

    @staticmethod
    def _mean_pooling(last_hidden_state, attention_mask):
        token_embeddings = last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, token_hidden_states, attention_mask, training_steps=True):
        """
        param: token_hidden_state: the token-level encoder_output hidden states
        """
        # pooled relevance-aware document embeddings 
        hidden_states = self._mean_pooling(token_hidden_states, attention_mask)
        BNM, H = hidden_states.shape
        
        # VAE's encode
        mean = self.hidden2mean(hidden_states)
        logv = self.hidden2logv(hidden_states)
        std = torch.exp(0.5 * logv)

        # VAE's decode
        is_train = (training_steps is not None)
        if is_train:
            r = torch.randn([BNM, self.latent_size], device=hidden_states.device)
            z = r * std + mean
            hidden_states_r = self.decoder(z)
            return (hidden_states_r[:, None, :],
                    mean.view(-1, self.latent_size), 
                    logv.view(-1, self.latent_size))
        else:
            z = mean
            hidden_states_r = self.decoder(z)
            return (hidden_states_r[:, None, :],)

