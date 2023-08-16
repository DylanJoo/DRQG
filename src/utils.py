import json
import random
import torch
from tqdm import tqdm
import math
import numpy as np
from torch.nn import CosineEmbeddingLoss, CrossEntropyLoss

def interpolate(A, B, n):
    return [torch.lerp(A, B, i) for i in np.linspace(0, 1, n)]

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

def sim_loss(a, b, metric='cosine'):
    loss_fct = CosineEmbeddingLoss()
    labels = [-1] * a.size(0) * a.size(1)
    loss = loss_fct(a.view(-1, a.size(-1)),
                    b.view(-1, b.size(-1)),
                    torch.tensor(labels).to(a.device))
    return loss

def kl_loss(logv1, mean1, logv2=None, mean2=None, reduction='sum'): # [batch_size(64), hidden_size(768)]
    if logv2 is None and mean2 is None:
        return -0.5 * torch.sum(1 + logv1 - mean1.pow(2) - logv1.exp())

    exponential = 1 + (logv1-logv2) - (mean1-mean2).pow(2)/logv2.exp() - (logv1-logv2).exp()
    kl_loss_embeds = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
    if reduction == 'sum':
        return kl_loss_embeds.sum()
    else:
        return kl_loss_embeds.mean()

def PairwiseCELoss(scores):
    CELoss = CrossEntropyLoss()
    logits = scores.view(2, -1).permute(1, 0) # (B*2 1) -> (B 2)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return CELoss(logits, labels)

def random_masking(tokens_lists, masked_token):

    for i, tokens_list in enumerate(tokens_lists):
        tokens = tokens_list.split()
        n_tokens = len(tokens)
        masked = random.sample(range(n_tokens), math.floor(n_tokens * 0.15))

        for j in masked:
            tokens[j] = masked_token

        tokens_list[i] = " ".join(tokens)

    return tokens_lists

def load_runs(path, output_score=False): 
    run_dict = collections.defaultdict(list)
    with open(path, 'r') as f:
        for line in tqdm(f):
            qid, _, docid, rank, score, _ = line.strip().split()
            run_dict[qid] += [(docid, float(rank), float(score))]

    sorted_run_dict = collections.OrderedDict()
    for (qid, doc_id_ranks) in tqdm(run_dict.items()):
        sorted_doc_id_ranks = \
                sorted(doc_id_ranks, key=lambda x: x[1], reverse=False) # score with descending order
        if output_score:
            sorted_run_dict[qid] = [(docid, rel_score) for docid, rel_rank, rel_score in sorted_doc_id_ranks]
        else:
            sorted_run_dict[qid] = [docid for docid, _, _ in sorted_doc_id_ranks]

    return sorted_run_dict

def transform_pred_to_good_read(path_jsonl, path_txt, sigma_map=None): 
    fr = open(path_jsonl, 'r')
    fw = open(path_txt, 'w')

    for line in tqdm(fr):
        data = json.loads(line.strip())
        fw.write(f"passage:\n{data.pop('passage')}\n")
        fw.write(f"groun truth: \n")
        fw.write(f"+\t{data.pop('positive_truth')[:2]}\n")
        fw.write(f"-\t{data.pop('negative_truth')[:2]}\n")

        for key in data:
            fw.write(f"{key}:\n")
            n_side = (len(data[key])-1) // 2
            if sigma_map is None:
                sigma_map = list(range(-n_side, n_side+1, 1))
            qlist = [f"{sigma_map[i]}\t{q}" for (i, q) in enumerate(data[key])]
            fw.write("\n".join(qlist))
            fw.write("\n\n")

    fr.close()
    fw.close()

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]
