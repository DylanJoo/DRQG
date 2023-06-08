import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple
from transformers.utils import ModelOutput

@dataclass
class Seq2SeqCVQGOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # reparameterized loss
    reparam_loss: Optional[torch.FloatTensor] = None
    # classification logit placeholder
    clf_logits: torch.FloatTensor = None
