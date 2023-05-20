import json
import random
import torch
from tqdm import tqdm
import math
import numpy as np
from torch.nn import CosineEmbeddingLoss, CrossEntropyLoss

def interpolate(A, B, n):
    return [torch.lerp(A, B, i) for i in np.linspace(0, 1, n)]

def kl_weight(annealing_fn, steps, k, x0):
    if annealing_fn == 'logistic':
        return float(1/(1+np.exp(-k*(steps-x0))))
    elif annealing_fn == 'linear':
        return min(1, steps/x0)
    else:
        return 1

def sim_loss(a, b, metric='cosine'):
    loss_fct = CosineEmbeddingLoss()
    labels = [-1] * a.size(0) * a.size(1)
    loss = loss_fct(a.view(-1, a.size(-1)),
                    b.view(-1, b.size(-1)),
                    torch.tensor(labels).to(a.device))
    return loss

def kl_loss(logv, mean):
    return -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

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

def transform_pred_to_good_read(path_jsonl, path_txt): 
    fr = open(path_jsonl, 'r')
    fw = open(path_txt, 'w')

    for line in tqdm(fr):
        data = json.loads(line.strip())
        fw.write(f"passage:\n{data.pop('passage')}\n")
        fw.write(f"groun truth: \n")
        fw.write(f"+\t{data.pop('positive_truth')}\n")
        fw.write(f"-\t{data.pop('negative_truth')}\n")

        for key in data:
            fw.write(f"{key}:\n")
            n_side = (len(data[key])-1) // 2
            sigma_map = list(range(-n_side, n_side+1, 1))
            qlist = [f"{sigma_map[i]}\t{q}" for (i, q) in enumerate(data[key])]
            fw.write("\n".join(qlist))
            fw.write("\n\n")

    fr.close()
    fw.close()

