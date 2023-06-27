import math
import torch
from torch import nn
from torch.nn import functional as F
import copy
from torch.nn import CrossEntropyLoss, KLDivLoss

def indoc_cont_loss(hidden_states, bs, norm=False):
    device = hidden_states.device
    if (hidden_states.size(1) != 1) or (len(hidden_state.shape)>2):
        hidden_state = hidden_states.mean(1)[:, None, :]
    else:
        hidden_state = hidden_states

    hs = hidden_state.size(-1)
    if norm:
        hidden_state = torch.nn.functional.normalize(hidden_state, p=2, dim=-1)

    ib_hidden_state = hidden_state.view(bs, -1, hs)
    # b n L H x b n H L 
    ib_scores = ib_hidden_state @ ib_hidden_state.transpose(-1, -2)
    loss_fct = CrossEntropyLoss()
    n_size = ib_scores.size(1)
    ib_labels = torch.arange(0, n_size, device=device)
    loss = loss_fct(
            ib_scores.view(-1, n_size), ib_labels.repeat(bs)
    )
    return loss

def pairwise_cont_loss(hidden_states, hidden_states_src=None, bs=1, norm=False):
    device = hidden_states.device
    hs = hidden_states.size(-1)
    ls = hidden_states.size(-2)

    if hidden_states_src is None:
        hidden_states_src = hidden_states

    if norm:
        hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
        hidden_states_src = torch.nn.functional.normalize(hidden_states_src, p=2, dim=-1)

    doc_hidden_state = hidden_states.view(bs, 2, -1, ls, hs)
    n=doc_hidden_state.size(1)*doc_hidden_state.size(2)
    doc_hidden_state_src = hidden_states_src.view(bs, n, -1, hs)

    # b 2 n ls hs --> b n ls hs
    doc_hidden_state_pos = doc_hidden_state[:, 0] 
    doc_hidden_state_neg = doc_hidden_state[:, 1]
    # b 2n ls hs --> b ls hs
    doc_hidden_state_src = doc_hidden_state_src[:, 0].unsqueeze(1)

    # pariwise logits
    # b n ls hs --> b n ls ls 
    doc_scores_pos = doc_hidden_state_src @ doc_hidden_state_pos.transpose(-1, -2)
    doc_scores_neg = doc_hidden_state_src @ doc_hidden_state_neg.transpose(-1, -2)

    # maxsim
    # b n ls ls --> b n --> bn 1
    doc_scores_pos = doc_scores_pos.max(-1).values.sum(-1).view(-1, 1)
    doc_scores_neg = doc_scores_neg.max(-1).values.sum(-1).view(-1, 1)

    # maxsim over the original doc
    # bn 2
    doc_scores = torch.cat([doc_scores_pos, doc_scores_neg], -1) 

    loss_fct = CrossEntropyLoss()
    loss = loss_fct(
            doc_scores, 
            torch.zeros(doc_scores.size(0), dtype=torch.long, device=device)
    )
    return loss

def gen_mle_loss(lm_logits, seq_labels, vocab_size):
    """ 
    [TODO] mask, reweight, contrastive framework
    """
    loss_gen_pos, loss_gen_neg = 0, 0
    loss_fct = CrossEntropyLoss(reduction='none')

    loss_gen_neg = loss_fct(
            lm_logits[seq_labels<1].view(-1, vocab_size), 
            labels[seq_labels<1].view(-1)
    ).mean()

    loss_gen_pos = loss_fct(
            lm_logits[seq_labels==1].view(-1, vocab_size), 
            labels[seq_labels==1].view(-1)
    ).mean()

    return {"pos": loss_gen_pos, "neg": loss_gen_neg}

def ql_kl_loss(clf_logits, clf_scores):
    loss_fct = KLDivLoss(reduction='sum')
    logp = F.log_softmax(clf_logits.view(-1, 2), -1) # BL 2
    target = torch.cat([(1-clf_scores).view(-1, 1), clf_scores.view(-1, 1)], -1)
    loss = loss_fct(logp, target)
    return loss / labels.size(0)
