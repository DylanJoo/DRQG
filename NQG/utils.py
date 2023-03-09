import random
import torch
import math
import numpy as np

def kl_weight(anneal_fn, step, k, x0):
    if anneal_fn == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

def kl_loss(logv, mean):
    """
    Parameters
    ----------
    logv: `torch.tensor`
        mapped batch embeddings with (B L H)
    mean: `torch.tensor`
        mapped batch embeddings with (B L H)
    """
    return -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

def hellinger_loss(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) /np.sqrt(2)

def hellinger_loss(p, q):
    return torch.sqrt(
            torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2)
    ) / np.sqrt(2)

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

# def load_collections(path=None, dir=None, candidate_set=None):
#     collection_dict = {}
#
#     if dir: # load if there are many jsonl files
#         files = [os.path.join(dir, f) for f in os.listdir(dir) if ".json" in f]
#     else:
#         files = [path]
#
#     for file in files:
#         print(f"Loading from collection {file}...")
#         with open(file, 'r') as f:
#             for i, line in enumerate(f):
#                 example = json.loads(line.strip())
#                 if candidate_set:
#                     if example['id'] in candidate_set:
#                         collection_dict[example['id']] = example['contents'].strip()
#                         candidate_set.remove(example['id'])
#                     if len(candidate_set) == 0:
#                         break
#                 else:
#                     collection_dict[example['id']] = example['contents'].strip()
#
#                 if i % 1000000 == 1:
#                     print(f" # documents...{i}")
#
#     print("DONE")
#     return collection_dict
