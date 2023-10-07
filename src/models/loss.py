import math
import torch
from torch import nn
from torch.nn import functional as F
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
        return {'pos': loss_gen_pos.mean()/L, 'neg': loss_gen_neg.mean()/L}
    else:
        return {'pos': loss_gen_pos, 'neg': loss_gen_neg}

def gen_mle_unloss(lm_logits, labels, seq_labels, average=True):
    lm_prob = torch.clamp( (1-lm_logits.softmax(-1)), min=1e-5)
    # lm_prob = torch.clamp( (-lm_logits).softmax(-1), min=1e-5 )
    lm_likelihood = lm_prob.log()
    loss_gen_pos, loss_gen_neg = 0, 0
    loss_fct = NLLLoss(reduction='none')
    B, L, V = lm_logits.shape

    if len(labels[seq_labels==1]) > 0:
        loss_gen_pos_from_neg = loss_fct(
                lm_likelihood[seq_labels==1].view(-1, V), 
                labels[seq_labels==1].view(-1)
        ).view(-1, L).sum(1)
    if len(labels[seq_labels<1]) > 0:
        loss_gen_neg_from_pos = loss_fct(
                lm_likelihood[seq_labels<1].view(-1, V), 
                labels[seq_labels<1].view(-1)
        ).view(-1, L).sum(1)

    if average:
        return {'neg2pos': loss_gen_pos_from_neg.mean()/L, 
                'pos2neg': loss_gen_neg_from_pos.mean()/L}
    else:
        return {'neg2pos': loss_gen_pos_from_neg, 'pos2neg': loss_gen_neg_from_pos}

def slic_margin_loss(logits_bar, logits_hat, maks_bar, mask_hat, seq_labels):
    bertscore = greedy_cos_idf(logits_bar, mask_bar, 
                               logits_hat, mask_hat)

    loss_f1_pos = bertscore[2][seq_labels==1].mean()
    loss_f1_neg = bertscore[2][seq_labels!=1].mean()
    return {'pos': loss_f1_pos, 'neg': loss_f1_neg}

def cosine_sim_loss(x, y, fn='cosine'):
    if fn == 'cosine':
        loss_fct = CosineEmbeddingLoss(margin=0.1, reduction='none')
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        target = torch.tensor([-1]).to(x.device)
        return loss_fct(x, y, target).mean()

def inbatch_cont_sim_loss(hidden_states, bs=1, norm=False):
    device = hidden_states.device
    hs = hidden_states.size(-1)
    if (hidden_states.size(1) != 1) or (len(hidden_states.shape)>2):
        hidden_state = hidden_states.mean(1)
    else:
        hidden_state = hidden_states

    if norm:
        hidden_state = F.normalize(hidden_state, p=2, dim=-1)

    hidden_state = hidden_state.view(-1, hs)
    # bs H x H bs
    indoc_scores = hidden_state @ hidden_state.permute(1, 0)
    loss_fct = CrossEntropyLoss(reduction='none')
    n_size = indoc_scores.size(0)
    indoc_labels = torch.arange(0, n_size, device=device)
    return loss_fct(indoc_scores, indoc_labels).mean() 

def greedy_cos_idf(ref_embedding, ref_masks, hyp_embedding, hyp_masks):

    batch_size = ref_embedding.size(0)

    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)
    masks = masks.float().to(sim.device)
    sim = sim * masks

    precision_scores, indices_precision = sim.max(dim=2)
    recall_scores, indices_recall = sim.max(dim=1)

    P = precision_scores.sum(dim=1)
    R = recall_scores.sum(dim=1)
    F1 = 2 * P * R / (P + R)
    F1 = F1.masked_fill(torch.isnan(F1), 0.)

    return P, R, F1

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

def ql_kl_loss(clf_logits, clf_scores):
    loss_fct = KLDivLoss(reduction='sum')
    logp = F.log_softmax(clf_logits.view(-1, 2), -1) # BL 2
    target = torch.cat([(1-clf_scores).view(-1, 1), clf_scores.view(-1, 1)], -1)
    loss = loss_fct(logp, target)
    return loss / clf_scores.size(0)


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
