import math
import torch
from torch import nn
from torch.nn import functional as F
import copy
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss

def indoc_cont_loss(hidden_states, bs=1, norm=False):
    device = hidden_states.device
    if (hidden_states.size(1) != 1) or (len(hidden_state.shape)>2):
        hidden_state = hidden_states.mean(1)[:, None, :]
    else:
        hidden_state = hidden_states

    hs = hidden_state.size(-1)
    if norm:
        hidden_state = torch.nn.functional.normalize(hidden_state, p=2, dim=-1)

    v_hidden_state = hidden_state.view(bs, -1, hs)
    # b n L H x b n H L 
    indoc_scores = v_hidden_state @ v_hidden_state.transpose(-1, -2)
    loss_fct = CrossEntropyLoss()
    n_size = indoc_scores.size(1)
    indoc_labels = torch.arange(0, n_size, device=device)
    loss = loss_fct(
            indoc_scores.view(-1, n_size), indoc_labels.repeat(bs)
    )
    return loss

def pairwise_cont_loss(hidden_states, hidden_base=None, bs=1, norm=False):
    """
    hidden_states: the perturbed query/document representation
    hidden_base: basic anchor embeddings, e.g., original document representation
    """
    device = hidden_states.device
    hs = hidden_states.size(-1)
    ls = hidden_states.size(-2)

    if hidden_base is None:
        hidden_base = hidden_states

    if norm:
        hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
        hidden_base = torch.nn.functional.normalize(hidden_base, p=2, dim=-1)

    # bsz 2 n l hsz
    reference = hidden_states.view(bs, 2, -1, ls, hs)
    n = reference.size(1)*reference.size(2)
    # bsz 2 n l hsz --> bsz n l hsz
    ref_pos = reference[:, 0] 
    ref_neg = reference[:, 1]

    # bsz*2n l' hs --> bsz 2n l' hs
    base = hidden_base.view(bs, n, -1, hs)
    # bsz (2n) l' hsz --> bsz 1 l' hsz
    base = base[:, 0].unsqueeze(1)

    # pariwise logits
    # bsz 1 l' hsz * bsz n hsz l --> b n l' l
    scores_pos = base @ ref_pos.transpose(-1, -2)
    scores_neg = base @ ref_neg.transpose(-1, -2)

    # maxsim
    # b n l' l --> b n l' --> b n  
    scores_pos = scores_pos.max(-1).values.sum(-1).view(-1, 1)
    scores_neg = scores_neg.max(-1).values.sum(-1).view(-1, 1)

    # maxsim over the original doc
    # bn 2
    scores = torch.cat([scores_pos, scores_neg], -1) 

    loss_fct = CrossEntropyLoss()
    loss = loss_fct(
            scores, 
            torch.zeros(scores.size(0), dtype=torch.long, device=device)
    )
    return loss

def gen_mle_loss(lm_logits, labels, seq_labels, vocab_size):
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
    return loss / clf_scores.size(0)

def gen_mle_gumbel_loss(lm_logits, labels, seq_labels, vocab_size, training_steps=0):
    loss_gen_pos, loss_gen_neg = 0, 0
    loss_fct = NLLLoss(reduction='none')
    tau_hp = max(0.5, math.exp(-1*1e-5*training_steps))
    logp_gumbel = F.gumbel_softmax(lm_logits, tau=tau_hp, hard=False)

    loss_gen_neg = loss_fct(
            logp_gumbel[seq_labels<1].log().view(-1, vocab_size),
            labels[seq_labels<1].view(-1)
    ).mean()

    loss_gen_pos = loss_fct(
            logp_gumbel[seq_labels==1].log().view(-1, vocab_size),
            labels[seq_labels==1].view(-1)
    ).mean()
    return {"pos": loss_gen_pos, "neg": loss_gen_neg}


# def indoc_kld_loss(hidden_states, hidden_states_src=None, bs=1):
#     device = hidden_states.device
#     if (hidden_states.size(1) != 1) or (len(hidden_state.shape)>2):
#         hidden_state = hidden_states.mean(1)[:, None, :]
#         hidden_state_src = hidden_states_src.mean(1)[:, None, :]
#     else:
#         hidden_state = hidden_states
#         hidden_state_src = hidden_states_src
#
#     hs = hidden_state.size(-1)
#     hidden_state = torch.nn.functional.normalize(hidden_state, p=2, dim=-1)
#     hidden_state_src = torch.nn.functional.normalize(hidden_state_src, p=2, dim=-1)
#
#     v_hidden_state = hidden_state.view(bs, -1, hs)
#     u_hidden_state = hidden_state_src.view(bs, -1, hs)
#     # b n L H x b n H L 
#     indoc_scores = u_hidden_state @ v_hidden_state.transpose(-1, -2)
#     loss_fct = CrossEntropyLoss()
#     n_size = indoc_scores.size(1)
#     indoc_labels = torch.arange(0, n_size, device=device)
#     loss = loss_fct(
#             indoc_scores.view(-1, n_size), indoc_labels.repeat(bs)
#     )
#     return loss
