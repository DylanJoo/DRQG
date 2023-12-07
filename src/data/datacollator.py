""" The datacollator for pcentric dataset.
"""
import string
import torch
import random
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

@dataclass
class DataCollatorBase:
    """ This datacollator is specified for evaluation process.
    While the initial arugments are also available for others in training.
    """
    tokenizer: Union[PreTrainedTokenizerBase] = None
    pad_to_multiple_of: Optional[int] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_p_length: Optional[int] = 512
    max_q_length: Optional[int] = 64
    return_tensors: str = "pt"
    prefix: Optional[str] = "{1}"
    scores: List[float] = None
    device: Optional[str] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        texts = [] 
        passages = [batch['passage'] for batch in features]

        for passage in passages:
            for score in self.scores:
                printed_score = round(score*100)
                texts.append(self.prefix.format(printed_score, passage))

        inputs = self.tokenizer(
                texts,
                max_length=self.max_p_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
        )

        return inputs, passages

    # def prepare_input(self, passage, queries, rel_scores, m=1):
    #
    #     texts_src = []
    #     texts_tgt = []
    #     scores = []
    #
    #     for i in range(m):
    #         j = i
    #         try:
    #             texts_tgt += [queries[j]]
    #             scores += [rel_scores[j]]
    #         except: 
    #             offset = int(j % len(queries))
    #             texts_tgt += [queries[offset]]
    #             scores += [rel_scores[offset]]
    #
    #         printed_score = round(scores[-1]*100)
    #         texts_src += [self.prefix.format(printed_score, passage)]
    #
    #     return texts_src, texts_tgt, scores

    def prepare_input(self, passage, queries, rel_scores, m=2, k=1):
        assert m >= k, f'm={m} is not larger than k={k}'
        # assert m <= len(queries), f'm needs to be smaller than {len(queries)}'
        assert len(queries) == len(rel_scores), f'lengths are not matched.'

        texts_src = []
        texts_tgt = []
        scores = []

        n_list = list(range( max( len(queries), m )))
        # n_list = list(range(m)
        if self.random:
            # min( (m-k), len(n_list[k:]) )
            m_list = n_list[:k] + sorted(random.sample(n_list[k:], k=m-k))
        else:
            m_list = n_list[:m]

        for i in m_list:
            try:
                texts_tgt += [queries[i]]
                scores += [rel_scores[i]]
            except: 
                offset = int(i % len(queries))
                texts_tgt += [queries[offset]]
                scores += [rel_scores[offset]]

            printed_score = round(scores[-1]*100)
            texts_src += [self.prefix.format(printed_score, passage)]

        return texts_src, texts_tgt, scores

@dataclass
class DataCollatorForBaseline(DataCollatorBase):
    m_negatives: int = 2
    m_positives: int = 2

    def __call__(self, 
                 features: List[Dict[str, Any]],
                 is_eval: Optional[bool] = False) -> Dict[str, Any]:

        if is_eval:
            return super().__call__(features)

        # text and id info 
        texts_src = []
        texts_tgt = []
        scores = []
        labels = []

        # collect passage to query
        for i, batch in enumerate(features):
            ## m positiive
            ## positive: 1, 0.9, 0.8 ...
            src1, tgt1, score1 = self.prepare_input(
                    batch['passage'], 
                    batch['positive'], 
                    batch['positive_score'],
                    m=self.m_positives,
                    k=self.k
            )
            ## m negative (reverse)
            ## negative: 0.2, 0.1, 0 --> 0, 0.1, 0.2 ... 
            src0, tgt0, score0 = self.prepare_input(
                    batch['passage'], 
                    batch['negative'][::-1], 
                    batch['negative_score'][::-1],
                    m=self.m_negatives,
                    k=self.k
            )
            ## change them into the same ordering
            ## since we will use the positive and negative pairwise loss
            ## align them with [top1, bottom2] and [top2, bottom1]
            src0, tgt0, score0 = src0[::-1], tgt0[::-1], score0[::-1]
            texts_src += src1 + src0
            texts_tgt += tgt1 + tgt0
            labels += [1]*len(src1) + [0]*len(src0)
            scores += score1 + score0

        # tokenization
        inputs = self.tokenizer(
                texts_src,
                max_length=self.max_p_length,
                truncation=True,
                padding=True,
                return_tensors=self.return_tensors,
        )
        targets = self.tokenizer(
                texts_tgt,
                max_length=self.max_q_length,
                truncation=True,
                padding=True,
                return_tensors=self.return_tensors
        )
        target_mask = targets['attention_mask'].bool()
        target_ids = targets['input_ids'].masked_fill(~target_mask, -100)
        inputs['labels'] = target_ids
        inputs['decoder_attention_mask'] = target_mask
        inputs['rel_labels'] = torch.Tensor(labels)
        inputs['rel_scores'] = torch.Tensor(scores)
        inputs['passage'] = features[0]['passage']
        return inputs


@dataclass
class DataCollatorForPromptQG(DataCollatorForBaseline):
    prompt_length: int = 0
    m_negatives: int = 2
    m_positives: int = 2
    random: bool = False
    k: int = 1

    def __call__(self, 
                 features: List[Dict[str, Any]], 
                 is_eval: Optional[bool] = False) -> Dict[str, Any]:

        if is_eval:
            inputs, passage = super().__call__(features, True)
            inputs['attention_mask'] = self._expand(inputs['attention_mask'])
            return inputs, passage
        else:
            inputs = super().__call__(features, False)
            inputs['attention_mask'] = self._expand(inputs['attention_mask'])
            return inputs

    def _expand(self, mask):
        additional_mask = torch.ones((mask.size(0), self.prompt_length))
        return torch.cat([additional_mask, mask], -1)
