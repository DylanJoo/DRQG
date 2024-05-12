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
    prefix: Optional[str] = "{0}"
    scores: List[float] = None
    device: Optional[str] = None
    random: bool = False
    k: int = 1

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        texts = [] 
        passages = [batch['passage'] for batch in features]

        for passage in passages:
            for score in self.scores:
                printed_score = round(score*100)
                texts.append(self.prefix.format(passage, printed_score))

        inputs = self.tokenizer(
                texts,
                max_length=self.max_p_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
        )

        return inputs, passages

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

            printed_score = round(scores[-1]*100) # round at 1 digit
            texts_src += [self.prefix.format(passage, printed_score)]

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
            )
            ## m negative (reverse)
            ## negative: 0.2, 0.1, 0 --> 0, 0.1, 0.2 ... 
            src0, tgt0, score0 = self.prepare_input(
                    batch['passage'], 
                    batch['negative'][::-1], 
                    batch['negative_score'][::-1],
                    m=self.m_negatives,
            )
            ## change them into the same ordering
            ## since we will use the positive and negative pairwise loss
            ## align them with [top1, bottom2] and [top2, bottom1] if m=2
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
    decoder_start_token_id: int = 0
    pad_token_id: int = 0
    corrupt_token_id: int = 0
    random_corrupt_rate: Optional[float] = None

    # this is clone from 'https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/t5/modeling_t5.py#L1523'
    def _random_and_shift_right(self, input_ids, rel_labels):
        decoder_start_token_id = self.decoder_start_token_id
        pad_token_id = self.pad_token_id
        corrupt_token_id = self.corrupt_token_id

        # random masking
        mask = torch.empty(input_ids.shape).bernoulli(1-self.random_corrupt_rate).bool()
        mask[(rel_labels == 1), :] = 1 # positive would not be masked
        mask[:, 0] = 1 # the first token would not be masked

        # new
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        shifted_input_ids.masked_fill_(~mask, corrupt_token_id)

        return shifted_input_ids

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

            # random mask the decoder input ids
            if self.random_corrupt_rate is not None:
                inputs['decoder_input_ids'] = self._random_and_shift_right(
                        input_ids=inputs['labels'],
                        rel_labels=inputs['rel_labels']
                )
            return inputs

    def _expand(self, mask):
        additional_mask = torch.ones((mask.size(0), self.prompt_length))
        return torch.cat([additional_mask, mask], -1)
