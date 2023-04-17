import random
import torch
import math
import numpy as np

def interpolate(A=None, B=None, n=3, tokenizer=None):
    """
    Params
    ------
    A: torch.Tensor
    B: torch.Tensor (B, H)
    """
    if A is None:
        A = tokenizer('<extra_id_10>', return_tensors='pt').to(B.device)
        A = torch.repeat((B.shape[0], 1))

    return [torch.lerp(A, B, i) for i in np.linspace(0, 1, n)]

def kl_weight(anneal_fn, step, k, x0):
    if anneal_fn == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

def kl_loss(logv, mean):
    return -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

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
