""" Copy the source code from BERTScore repo.  """
import sys
import os
import torch

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

