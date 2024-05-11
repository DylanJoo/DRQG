import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
# module
from .encoder import GTREncoder, GTEEncoder
from .readgen import READGen as generator
# root
from utils import batch_iterator

class READEval:
    def __init__(
        self, 
        dataset=None,
        encoder_name=None,
        regressor_name=None,
        reranker_name=None,
        device='cuda',
        generator=None, 
    ):
        self.dataset = dataset
        self.device = device
        self.generator = generator

        # diversity
        if encoder_name:
            if 'gte' in encoder_name.lower():
                self.encoder = GTEEncoder.from_pretrained(encoder_name)
            else:
                self.encoder = GTREncoder.from_pretrained(encoder_name)
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
            self.encoder.eval()
            self.encoder.to(device)
        ## [NOTE] Can also try one-embedder with prompts

        # ranking for regression (crossencoder)
        if regressor_name:
            self.regressor = AutoModelForSequenceClassification.from_pretrained(regressor_name)
            self.regressor_tokenizer = AutoTokenizer.from_pretrained(regressor_name)
            self.regressor.eval()
            self.regressor.to(device)

        # ranking for reranking (crossencoder)
        if reranker_name:
            self.reranker = T5ForConditionalGeneration.from_pretrained(reranker_name)
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_name)
            self.reranker.eval()
            self.reranker.to(self.device)

        # evaluation
        self.scores = []

    @torch.no_grad()
    def get_relevance_probs(
        self, 
        qtexts, 
        ptexts, 
        return_probs=False
    ):
        """ Followed the instructions in monoT5 paper """
        features = self.reranker_tokenizer(
                [f'Query: {q} Document: {p}' for q, p in zip(qtexts, ptexts)], 
                ['Relevant: </s>'] * len(qtexts),
                padding=True, 
                truncation='only_first', 
                add_special_tokens=False,
                return_tensors="pt"
        ).to(self.device)

        dummy = torch.full(
                features.input_ids.size(), 
                self.reranker.config.decoder_start_token_id
        ).to(self.device)

        result = self.reranker(**features, decoder_input_ids=dummy).logits
        result = result[:, 0, (1176, 6136)]
        return F.softmax(result, dim=-1)[:, 0].flatten().cpu().numpy()

    @torch.no_grad()
    def get_logit_scores(
        self, 
        qtexts, 
        ptexts, 
        return_logit=False
    ):
        """ Followed the instructions in sbert repo.  """
        features = self.regressor_tokenizer(
                qtexts, ptexts,
                padding=True, 
                truncation='only_second', 
                return_tensors="pt"
        ).to(self.device)
        scores = self.regressor(**features).logits
        return scores.flatten().cpu().numpy()

    @torch.no_grad()
    def get_embeddings(
        self, 
        qtexts, 
        return_distance=False
    ):
        """ Followed the instruction in sbert repo.  """
        features = self.encoder_tokenizer(
                qtexts,
                padding=True,
                truncation=True,
                return_tensors='pt'
        ).to(self.device)
        embeddings = self.encoder.encode(**features)
        return embeddings.detach().cpu().numpy()

    def evaluate_relevancy(
        self, 
        total_query_group, 
        total_passages, 
        total_scores, 
        batch_size=1, 
        select_score=1.0,
        **kwargs
    ):
        """ the reranker should be a pointwise reranker with probs (or the one has finite scoring)"""
        N = len(total_query_group)
        scores = {f"rel-{select_score}": []}
        group_boundaries = []
        count = 0

        # batch encoding
        for start, end in tqdm(
                batch_iterator(total_query_group, batch_size, True),
                total=N//batch_size + 1,
        ):
            batch_query_group = total_query_group[start:end]
            batch_passages = total_passages[start:end]
            batch_scores = total_scores[start:end]

            queries = []
            passages = []

            for i in range(len(batch_query_group)):
                # select a few of queries
                query_group = batch_query_group[i]
                passage = batch_passages[i]
                score_group = batch_scores[i]

                for query, score in zip(query_group, score_group):
                    if select_score == score:
                        queries += [query]
                        passages += [passage]
                        count += 1

            if len(queries) > 0:
                ## get relevance scores (prob value) of selected
                prob_scores = self.get_relevance_probs(queries, passages)
                scores[f"rel-{select_score}"].extend(prob_scores)

        return scores

    def evaluate_consistency(
        self, 
        total_query_group, 
        total_passages, 
        total_scores, 
        batch_size=1, 
        **kwargs
    ):
        N = len(total_query_group)
        total_scores = np.array(total_scores).flatten()
        all_logit_scores = []
        group_boundaries = []
        count = 0

        # batch encoding
        for start, end in tqdm(
                batch_iterator(total_query_group, batch_size, True),
                total=N//batch_size + 1,
        ):
            batch_query_group = total_query_group[start:end]
            batch_passages = total_passages[start:end]

            queries = []
            passages = []
            for i, query_group in enumerate(batch_query_group):
                group_boundaries.append( 
                        (count, count+len(query_group)) 
                )
                queries += query_group
                passages += [batch_passages[i]] * len(query_group)
                count += len(query_group)

            ## get relevance scores (logit value)
            logit_scores = self.get_logit_scores(
                    queries, passages
            )
            all_logit_scores.extend(logit_scores)

        correlations = {'pearson': [], 'spearman': []}
        for i in range(N):
            start, end = group_boundaries[i]
            logit_scores = all_logit_scores[start:end]
            relevances = total_scores[start:end]

            pearsonr = stats.pearsonr(logit_scores, relevances).statistic
            spearmanr = stats.spearmanr(logit_scores, relevances).statistic

            correlations['pearson'].append(
                    pearsonr if pearsonr is not np.nan else 0
            )
            correlations['spearman'].append(
                    spearmanr if spearmanr is not np.nan else 0
            )

        return correlations

    def evaluate_diversity(
        self, 
        total_query_group, 
        metrics=('euclidean', 'angular'),
        batch_size=1, 
        **kwargs
    ):
        N = len(total_query_group)
        group_boundaries = []
        all_embeddings = []
        count = 0

        # batch encoding
        for batch_query_group in tqdm(
                batch_iterator(total_query_group, batch_size),
                total=N//batch_size + 1
        ):
            queries = []
            for query_group in batch_query_group:
                group_boundaries.append( (count, count+len(query_group)) )
                queries += query_group
                count += len(query_group)

            ## get embeddings
            embeddings = self.get_embeddings(queries)
            all_embeddings.extend(embeddings)

        distances = {k: [] for k in metrics}
        for i in range(N):
            start, end = group_boundaries[i]
            embeddings = np.array(all_embeddings[start:end])

            if 'angular' in metrics:
                distances['angular'].append(
                        self.calculate_pairwise_angular(embeddings)
                )
            if 'euclidean' in metrics:
                distances['euclidean'].append(
                        self.calculate_euclidean(embeddings)
                )

        return distances

    @staticmethod
    def calculate_euclidean(xs):
        """ An example of "euclidean distance" and (l2)-norm
        > b = np.random.multivariate_normal([1,1,1], np.eye(3)*0.1, 10).T
        > np.linalg.norm(b - np.mean(b, 0), axis=1)
        array([0.65276676, 0.70749294, 0.65661753])
        > b = np.random.multivariate_normal([1,1,1], np.eye(3)*10, 10).T
        > np.linalg.norm(b - np.mean(b, 0), axis=1)
        array([ 8.48881862,  9.1476951 , 13.73675615])
        """
        xc = np.mean(xs, 0)
        return np.linalg.norm(xs-xc, axis=1).mean().tolist()

    @staticmethod
    def calculate_pairwise_angular(xs):
        """ An example of "angular distance" and "cosine similarity"
        > (np.arccos(cosine_similarity(a, b))/np.pi).flatten().argsort()
        array([4, 3, 5, 7, 8, 2, 6, 0, 1])
        > cosine_similarity(a, b).flatten().argsort()
        array([1, 0, 6, 2, 8, 7, 5, 3, 4])
        """
        N = xs.shape[0]
        cosine_matrix = np.dot(xs, xs.T) # they have been normalized
        cosine_matrix = np.clip(cosine_matrix, -1.0, 1.0)
        # angular_sim_matrix = 1 - angular_dist_matrix 
        angular_dist_matrix = np.arccos(cosine_matrix) / np.pi 
        # get the upper triangle (without diagonal)
        return angular_dist_matrix[np.triu_indices(N, k=1)].mean().tolist()

    # def dis_diversity(self, data, prefix, batch_size):
    #     # distribution/distant-based diversity
    #
    #     def get_mask_data(data_list, prefix_list):
    #         # mask the prefix and generated result respectively
    #         src_list, tgt_list, len_list = [], [], []
    #         for data_ele, prefix_ele in zip(data_list, prefix_list):
    #             assert data_ele.index(prefix_ele) == 0
    #             src_list_ele = [prefix_ele + ' <mask_1>', '<mask_1> ' + data_ele[len(prefix_ele):]]
    #             tgt_list_ele = [data_ele[len(prefix_ele):], prefix_ele]
    #             src_list.extend(src_list_ele)
    #             tgt_list.extend(tgt_list_ele)
    #             len_list.append(2)
    #         return src_list, tgt_list, len_list
    #
    #     src_data, tgt_data, data_len = get_mask_data(data, prefix)
    #
    #     # eval_score: score of each pattern evaluator
    #     # beta: (unnormalized) weight factor of each pattern evaluator
    #     eval_score, beta = [], []
    #     for data_id in tqdm(range(0, len(src_data), batch_size)):
    #         src_text, tgt_text = src_data[data_id: data_id + batch_size], tgt_data[data_id: data_id + batch_size]
    #         self.model.eval()
    #         with torch.no_grad():
    #             loss, tgt_score = self.lm_score(src_text, tgt_text, add_special_tokens=False)
    #             cur_score = [-loss_ele.detach().cpu().numpy() for loss_ele in loss]
    #
    #         eval_score.extend(cur_score)
    #         beta.extend(tgt_score)
    #
    #     # compute final score via the weighted sum of pattern evaluators
    #     data_st = 0
    #     res_score = []
    #     for len_ele in data_len:
    #         if sum(beta[data_st: data_st + len_ele]) > 0:
    #             res_score.append(np.dot(eval_score[data_st: data_st + len_ele], beta[data_st: data_st + len_ele]) /
    #                              sum(beta[data_st: data_st + len_ele]))
    #         else:
    #             res_score.append(np.mean(eval_score[data_st: data_st + len_ele]))
    #         data_st += len_ele
    #
    #     return res_score

    # def rel_consistency(self, data, label_str, batch_size):
    #     # attribute relevance
    #     label = [self.label_name.index(label_ele) for label_ele in label_str]
    #
    #     def get_mask_data(data_list, prompt_list, verbal_list):
    #         # use prompts and verbalizers to generate data
    #         src_list, tgt_list, len_list = [], [], []
    #         for data_ele in data_list:
    #             src_list_ele, tgt_list_ele = [], []
    #             for idx in range(len(prompt_list)):
    #                 for idy in range(len(verbal_list)):
    #                     for idz in range(len(verbal_list[0])):
    #                         src_list_ele.append(prompt_list[idx].replace('<gen_result>',
    #                                                                      data_ele).replace('<mask_token>', '<mask_1>'))
    #                         tgt_list_ele.append(verbal_list[idy][idz])
    #             src_list.extend(src_list_ele)
    #             tgt_list.extend(tgt_list_ele)
    #         return src_list, tgt_list

    #     src_data, tgt_data = get_mask_data(data, self.prompt_list, self.verbal_list)
    #
    #     # eval_score: LM score for each pair of prompts and verbalizers
    #     eval_score, beta = [], []
    #     for data_id in tqdm(range(0, len(src_data), batch_size)):
    #         src_text, tgt_text = src_data[data_id: data_id + batch_size], tgt_data[data_id: data_id + batch_size]
    #         self.model.eval()
    #         with torch.no_grad():
    #             loss, _ = self.lm_score(src_text, tgt_text, has_iwf=False, add_special_tokens=False)
    #             cur_score = [torch.exp(-loss_ele).detach().cpu().numpy() for loss_ele in loss]
    #
    #         eval_score.extend(cur_score)
    #
    #     score_pair = np.reshape(eval_score, (-1, len(self.verbal_list[0])))
    #     # compute unnormalized weight scores
    #     weight_unnormal = np.sum(score_pair, axis=1)
    #     # compute the score of each pattern evaluator
    #     score_pair /= np.sum(score_pair, axis=1, keepdims=True)
    #     score_data = np.reshape(score_pair, (-1, len(self.prompt_list) * len(self.verbal_list), len(self.verbal_list[0])))
    #     weight_unnormal = np.reshape(weight_unnormal, (-1, len(self.prompt_list) * len(self.verbal_list)))
    #     # compute normalized weight scores
    #     weight_normal = weight_unnormal / np.sum(weight_unnormal, axis=1, keepdims=True)
    #     weight_normal = np.expand_dims(weight_normal, axis=2)
    #     res_score = np.choose(np.array(label), np.sum(score_data * weight_normal, axis=1).T)
    #
    #     return res_score
    #
    # def score(self, aspect, data, prefix=None, label=None, batch_size=1):
    #     # aspect: coh (coherence), cons (consistency), or ar (attribute relevance)
    #     # data: list of generated texts
    #     # prefix: list of content prefixes
    #     # label: list of attribute labels
    #     if aspect == 'coh':
    #         return self.coh_score(data, batch_size)
    #     else:
    #         if aspect == 'cons':
    #             return self.cons_score(data, prefix, batch_size)
    #         else:
    #             return self.ar_score(data, label, batch_size)

    # def lm_score(self, src_text, tgt_text, has_iwf=True, add_special_tokens=True):
    #     # compute the log probability of pre-trained models
    #     batch = self.tokenizer(src_text, truncation=True, padding='longest',
    #                            return_tensors="pt").to(self.device)
    #     labels = self.tokenizer(tgt_text, truncation=True, padding='longest', add_special_tokens=add_special_tokens,
    #                             return_tensors="pt").to(self.device)
    #
    #     # use IWF scores as weights for coherence and consistency
    #     if has_iwf:
    #         tgt_score = [max([self.iwf_score[token_id] for token_id in
    #                           labels['input_ids'][label_id].cpu().numpy()]) for label_id in
    #                      range(labels['input_ids'].shape[0])]
    #     else:
    #         tgt_score = []
    #
    #     output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
    #                         labels=labels['input_ids'])
    #     logits = output.logits.view(-1, self.model.config.vocab_size)
    #     loss = self.loss_fct(logits, labels['input_ids'].view(-1))
    #     tgt_len = labels['attention_mask'].sum(dim=1)
    #     loss = loss.view(labels['input_ids'].shape[0], -1)
    #     loss = loss.sum(dim=1) / tgt_len
    #
    #     return loss, tgt_score

    # def coh_score(self, data, batch_size):
    #     # coherence
    #     data_split = [sent_tokenize(data_ele) for data_ele in data]
    #
    #     def get_mask_data(data_list):
    #         # mask each sentence respectively
    #         src_list, tgt_list, len_list = [], [], []
    #         for data_ele in data_list:
    #             src_list_ele, tgt_list_ele = [], []
    #             for idx in range(len(data_ele)):
    #                 tgt_list_ele.append(data_ele[idx])
    #                 src_list_ele.append(' '.join(data_ele[:idx]) + ' <mask_1> ' + ' '.join(data_ele[idx + 1:]))
    #             src_list.extend(src_list_ele)
    #             tgt_list.extend(tgt_list_ele)
    #             len_list.append(len(data_ele))
    #         return src_list, tgt_list, len_list
    #
    #     # data_len: list of the number of sentences in each generated result
    #     src_data, tgt_data, data_len = get_mask_data(data_split)
    #
    #     # eval_score: score of each pattern evaluator
    #     # beta: (unnormalized) weight factor of each pattern evaluator
    #     eval_score, beta = [], []
    #     for data_id in tqdm(range(0, len(src_data), batch_size)):
    #         src_text, tgt_text = src_data[data_id: data_id + batch_size], tgt_data[data_id: data_id + batch_size]
    #         self.model.eval()
    #         with torch.no_grad():
    #             loss, tgt_score = self.lm_score(src_text, tgt_text)
    #             cur_score = [-loss_ele.detach().cpu().numpy() for loss_ele in loss]
    #
    #         eval_score.extend(cur_score)
    #         beta.extend(tgt_score)
    #
    #     # compute final score via the weighted sum of pattern evaluators
    #     data_st = 0
    #     res_score = []
    #     for len_ele in data_len:
    #         if sum(beta[data_st: data_st + len_ele]) > 0:
    #             res_score.append(np.dot(eval_score[data_st: data_st + len_ele], beta[data_st: data_st + len_ele]) /
    #                              sum(beta[data_st: data_st + len_ele]))
    #         else:
    #             res_score.append(np.mean(eval_score[data_st: data_st + len_ele]))
    #         data_st += len_ele
    #
    #     return res_score

