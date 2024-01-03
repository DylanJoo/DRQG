import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import copy
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss, CosineEmbeddingLoss

def gen_mle_loss(lm_logits, labels, seq_labels, average=True):
    loss_fct = CrossEntropyLoss(reduction='none')
    B, L, V = lm_logits.shape

    if len(labels[seq_labels==1]) > 0:
        loss_gen_pos = loss_fct(
                lm_logits[seq_labels==1].view(-1, V), 
                labels[seq_labels==1].view(-1)
        ).view(-1, L).sum(1)

    if len(labels[seq_labels<1]) > 0:
        loss_gen_neg = loss_fct(
                lm_logits[seq_labels<1].view(-1, V), 
                labels[seq_labels<1].view(-1)
        ).view(-1, L).sum(1)

    if average:
        return {'pos': loss_gen_pos.mean()/L, 
                'neg': loss_gen_neg.mean()/L}
    else:
        return {'pos': loss_gen_pos, 
                'neg': loss_gen_neg}

# Unlikelihood training 
## adapt to calibration #1): maximized margin gap using generation probabilty
def gen_mle_unloss(lm_logits, labels, seq_labels, average=True):
    lm_prob = torch.clamp( (1-lm_logits.softmax(-1)), min=1e-5)
    # lm_prob = torch.clamp( (-lm_logits).softmax(-1), min=1e-5 )
    lm_logits = lm_prob.log()
    loss_gen_pos, loss_gen_neg = 0, 0
    loss_fct = NLLLoss(reduction='none')
    B, L, V = lm_logits.shape

    if len(labels[seq_labels==1]) > 0:
        loss_gen_pos = loss_fct(
                lm_logits[seq_labels==1].view(-1, V), 
                labels[seq_labels==1].view(-1)
        ).view(-1, L).sum(1)
    if len(labels[seq_labels<1]) > 0:
        loss_gen_neg = loss_fct(
                lm_logits[seq_labels<1].view(-1, V), 
                labels[seq_labels<1].view(-1)
        ).view(-1, L).sum(1)

    if average:
        return {'pos': loss_gen_pos.mean()/L, 
                'neg': loss_gen_neg.mean()/L}
    else:
        return {'pos': loss_gen_pos, 
                'neg': loss_gen_neg}

def slic_margin_loss(embeds_bar, embeds_hat, mask_bar, mask_hat, seq_labels, measurement='f1', ngrams=[1]):
    m = {'precision': 0, 'recall': 1, 'f1': 2}[measurement]
    loss_f1_pos = greedy_cos_idf(
            embeds_bar[seq_labels==1], 
            mask_bar[seq_labels==1],
            embeds_hat[seq_labels==1], 
            mask_hat[seq_labels==1],
            ngrams
    )[m].detach()

    loss_f1_neg = greedy_cos_idf(
            embeds_bar[seq_labels!=1], 
            mask_bar[seq_labels!=1],
            embeds_hat[seq_labels!=1], 
            mask_hat[seq_labels!=1],
            ngrams
    )[m].detach()

    return {'pos': loss_f1_pos, 'neg': loss_f1_neg}

# ## [NOTE] this function has no `idf` setups.
def greedy_cos_idf(ref_embedding, ref_masks, hyp_embedding, hyp_masks, ngrams):
    batch_size = ref_embedding.size(0)

    # inplace functions
    # ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    # hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
    ref_embedding = torch.div(ref_embedding, torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding = torch.div(hyp_embedding, torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)
    masks = masks.float().to(sim.device)
    sim = sim * masks

    bs = sim.shape[0]
    P, R = torch.zeros(bs, device=sim.device), torch.zeros(bs, device=sim.device)
    F1 = torch.zeros(bs, device=sim.device)
    for n in ngrams:
        n = int(n)
        if n==1:
            # based on hyp_embedding, ref_embedding
            precision_scores, indices_precision = sim.max(dim=2) 
            recall_scores, indices_recall = sim.max(dim=1)
        else: 
            n_ = min(sim.size(-2), n)
            pooler = nn.MaxPool2d((n_, sim.size(-1)), stride=(1,1))
            precision_scores = pooler(sim).squeeze()
            n_ = min(sim.size(-1), n)
            pooler = nn.MaxPool2d((sim.size(-2), n_), stride=(1,1))
            recall_scores = pooler(sim).squeeze()

        ## [NOTE] It's mean here.
        # p = precision_scores.mean(dim=1).flatten() * sim.size(-2)
        # r = recall_scores.mean(dim=1).flatten() * sim.size(-1)
        p = precision_scores.mean(dim=1).flatten() 
        r = recall_scores.mean(dim=1).flatten() 

        P += p
        R += r
        F1 += 2 * p * r / (1e-5 + p + r)

    # inplace functions
    # F1 = F1.masked_fill(torch.isnan(F1), 0.)

    return P, R, F1

def cosine_sim_loss(x, y):
    loss_fct = CosineEmbeddingLoss(margin=0.1, reduction='none')
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    target = torch.tensor([-1]).to(x.device)
    return loss_fct(x, y, target).mean()

def average_pooling(hidden_states, attention_mask=None):
    if (hidden_states.size(1) != 1) or (len(hidden_states.shape)>2):
        if attention_mask is not None:
            # print(hidden_states.shape)
            # print(attention_mask.shape)
            hidden_state = torch.mean(hidden_states * attention_mask.unsqueeze(-1), dim=1)
        else:
            hidden_state = torch.mean(hidden_states, dim=1)
    return hidden_state

def inbatch_cont_dd_sim_loss(
    hidden_states, 
    attention_mask=None, 
    bs=1, 
    norm=False, 
    reduction=None, 
    temperature=1,
    document_wise=False,
    relevance_wise=False
):
    device = hidden_states.device
    BNM, L, H = hidden_states.shape
    hidden_state = average_pooling(hidden_states, attention_mask)
    if norm:
        hidden_state = F.normalize(hidden_state, p=2, dim=-1)
    loss = 0
    loss_fct = CrossEntropyLoss(reduction='none')

    if document_wise: # document-wise (batch)
        # indoc: B N H x B H N
        hidden_state = hidden_state.view(bs, BNM//bs, H) / temperature
        inbatch_scores = hidden_state @ hidden_state.transpose(-1, -2)
        inbatch_scores = inbatch_scores.view(-1, BNM//bs)
        inbatch_labels = torch.arange(0, BNM//bs, device=device).repeat(bs)
        loss += loss_fct(inbatch_scores, inbatch_labels).mean()
    if relevance_wise: # query-wise 
        # indoc: B N H x B H N --> N B H x N H B
        hidden_state = hidden_state.view(bs, BNM//bs, H) / temperature
        hidden_state = hidden_state.permute(1, 0, 2)
        inbatch_scores = hidden_state @ hidden_state.transpose(-1, -2)
        inbatch_scores = inbatch_scores.view(-1, bs)
        inbatch_labels = torch.arange(0, bs, device=device).repeat(BNM//bs)
        loss += loss_fct(inbatch_scores, inbatch_labels).mean()
    if (document_wise is False) and (relevance_wise is False):
        # inbatch: BNM H x H BNM
        hidden_state = hidden_state.view(-1, H) / temperature
        inbatch_scores = hidden_state @ hidden_state.transpose(-1, -2)
        inbatch_labels = torch.arange(0, BNM, device=device)
        loss += loss_fct(inbatch_scores, inbatch_labels).mean()

    return loss

def inbatch_cont_qd_sim_loss(
    hidden_states, 
    q_hidden_states, 
    attention_mask=None,
    q_attention_mask=None,
    bs=1, 
    norm=False, 
    reduction=None, 
    temperature=1,
    document_wise=False,
    relevance_wise=False
):
    device = hidden_states.device
    BNM, L, H = hidden_states.shape
    hidden_state = average_pooling(hidden_states, attention_mask)
    q_hidden_state = average_pooling(q_hidden_states, q_attention_mask)
    if norm:
        hidden_state = F.normalize(hidden_state, p=2, dim=-1)
        q_hidden_state = F.normalize(q_hidden_state, p=2, dim=-1)

    loss = 0
    loss_fct = CrossEntropyLoss(reduction='none')

    if document_wise: # document-wise (batch)
        # indoc: B N H x B H N
        hidden_state = hidden_state.view(bs, BNM//bs, H) / temperature
        q_hidden_state = q_hidden_state.view(bs, BNM//bs, H) / temperature
        inbatch_scores = hidden_state @ q_hidden_state.transpose(-1, -2)
        inbatch_scores = inbatch_scores.view(-1, BNM//bs)
        inbatch_labels = torch.arange(0, BNM//bs, device=device).repeat(bs)
        loss += loss_fct(inbatch_scores, inbatch_labels).mean()
    if relevance_wise: # query-wise 
        # indoc: B N H x B H N --> N B H x N H B
        hidden_state = hidden_state.view(bs, BNM//bs, H) / temperature
        q_hidden_state = q_hidden_state.view(bs, BNM//bs, H) / temperature
        hidden_state = hidden_state.permute(1, 0, 2)
        q_hidden_state = q_hidden_state.permute(1, 0, 2)
        inbatch_scores = hidden_state @ q_hidden_state.transpose(-1, -2)
        inbatch_scores = inbatch_scores.view(-1, bs)
        inbatch_labels = torch.arange(0, bs, device=device).repeat(BNM//bs)
        loss += loss_fct(inbatch_scores, inbatch_labels).mean()
    if (document_wise is False) and (relevance_wise is False):
        # inbatch: BNM H x H BNM
        hidden_state = hidden_state.view(-1, H) / temperature
        q_hidden_state = q_hidden_state.view(-1, H) / temperature
        inbatch_scores = hidden_state @ q_hidden_state.transpose(-1, -2)
        inbatch_labels = torch.arange(0, BNM, device=device)
        loss += loss_fct(inbatch_scores, inbatch_labels).mean()

    return loss

def kl_weight(annealing_fn, steps, k=None, x0=None, n_total_iter=None, n_cycle=None):
    if steps is None:
        return 1
    if annealing_fn == 'logistic':
        return float(1/(1+np.exp(-k*(steps-x0))))
    elif annealing_fn == 'linear':
        return min(1, steps/x0)
    elif annealing_fn == 'cyclic':
        return frange_cycle_linear(n_total_iter, steps, n_cycle)

def frange_cycle_linear(n_total, curr, start=0.0, stop=1.0, n_cycle=4, ratio=1.0):
    L = np.ones(n_total) * stop
    period = n_total/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule
    return min(stop, start + step * (curr % period))

def ql_kl_loss(clf_logits, clf_score):
    loss_fct = KLDivLoss(reduction='sum')
    logp = F.log_softmax(clf_logits.view(-1, 2), -1) # BL 2
    target = torch.cat([(1-clf_scores).view(-1, 1), clf_scores.view(-1, 1)], -1)
    loss = loss_fct(logp, target)
    return loss / clf_scores.size(0)

def kl_loss(logv1, mean1, logv2=None, mean2=None, reduction='sum'): 
    # [batch_size(64), hidden_size(768)]
    if logv2 is None and mean2 is None:
        return -0.5 * torch.sum(1 + logv1 - mean1.pow(2) - logv1.exp())

    exponential = 1 + (logv1-logv2) - (mean1-mean2).pow(2)/logv2.exp() - (logv1-logv2).exp()
    loss = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
    if reduction:
        return loss.mean()
    else:
        return loss

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

# def gen_mle_gumbel_loss(lm_logits, labels, seq_labels, vocab_size, training_steps=0):
#     loss_gen_pos, loss_gen_neg = 0, 0
#     loss_fct = NLLLoss(reduction='none')
#     tau_hp = max(0.5, math.exp(-1*1e-5*training_steps))
#     logp_gumbel = F.gumbel_softmax(lm_logits, tau=tau_hp, hard=False)
#
#     loss_gen_neg = loss_fct(
#             logp_gumbel[seq_labels<1].log().view(-1, vocab_size),
#             labels[seq_labels<1].view(-1)
#     ).mean()
#
#     loss_gen_pos = loss_fct(
#             logp_gumbel[seq_labels==1].log().view(-1, vocab_size),
#             labels[seq_labels==1].view(-1)
#     ).mean()
#     return {"pos": loss_gen_pos, "neg": loss_gen_neg}

# def pairwise_maxsim_loss(hidden_states, bs=1, ms=2, norm=False):
#     device = hidden_states.device
#     hs = hidden_states.size(-1)
#     ls = hidden_states.size(-2)
#     if norm:
#         hidden_states = F.normalize(hidden_states, p=2, dim=2)
#
#     # reshape (bs, 2(pos/neg), ms, hs) -> reshape(2, bs, ms, ls, hs)
#     hidden_states = hidden_states.view(bs, 2, ms, ls, hs).permute(1, 0, 2, 3,4)
#     pos_hidden_states = hidden_states[0].reshape(-1, ls, hs)
#     pos_hidden_states = pos_hidden_states.repeat(2, 1, 1).contiguous()
#     base_hidden_states = hidden_states.reshape(-1, ls, hs)
#
#     # (2bsms ls hs) x (2bsms ls hs) = (2bsms ls ls) = (2bsms ls) = (2bsms, 0)
#     pairwise_scores = (pos_hidden_states @ base_hidden_states.permute(
#             0, 2, 1)).max(2).values.sum(1)
#
#     # multiplication (bs*ms, bs*ms*2)
#     loss_fct = CrossEntropyLoss(reduction='none')
#     pairwise_scores = pairwise_scores.view(2, -1).permute(1, 0) 
#     pairwise_labels = torch.zeros(pairwise_scores.size(0), dtype=torch.long, device=device)
#     return loss_fct(pairwise_scores, pairwise_labels).mean()

