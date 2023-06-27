""" The datacollator for pcentric dataset.
"""
import torch
import random
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

@dataclass
class DataCollatorBase:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    pad_to_multiple_of: Optional[int] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_p_length: Optional[int] = 512
    max_q_length: Optional[int] = 64
    return_tensors: str = "pt"
    is_eval: Union[bool] = False
    prefix: str = ""
    irrelevant_included: bool = field(default=False)
    relevant_included: bool = field(default=True)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        texts_p = [batch['passage'] for batch in features]
        texts_q1 = [batch['positive'] for batch in features]
        texts_q0 = [batch['negative'] for batch in features]

        if self.irrelevant_included and self.relevant_included:
            texts_q = [random.sample((q1, q0), k=1)[0] \
                    for q1, q0 in zip(texts_q1, texts_q0)]
        else:
            if self.irrelevant_included:
                texts_q = texts_q0
            if self.relevant_included:
                texts_q = texts_q1

        inputs = self.tokenizer(
                [f"{self.prefix}{p}" for p in texts_p],
                max_length=self.max_p_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
        )
        target_ids = self.tokenizer(
                texts_q,
                max_length=self.max_q_length,
                padding=True,
                return_tensors='pt'
        ).input_ids
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = target_ids

        if self.is_eval:
            inputs['passage'] = texts_p
            inputs['positive'] = texts_q1
            inputs['negative'] = texts_q0

        return inputs


@dataclass
class DataCollatorForVQG(DataCollatorBase):
    is_train: Union[bool, str] = False
    is_eval: Union[bool, str] = False
    m_negatives: int = 2
    m_positives: int = 2
    use_clf_score: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # text and id info 
        texts_p = []
        texts_q = []
        clf_labels = []
        clf_scores = []
        batch_id = []

        for i, batch in enumerate(features):

            p = batch['passage']
            batch_pos = batch['positive']
            batch_pos_scores = batch['positive_score']
            batch_neg = batch['negative']
            batch_neg_scores = batch['negative_score']

            for j in range(self.m_positives):
                texts_p += [p]
                clf_labels += [1]
                try:
                    texts_q += [batch_pos[j]]
                    clf_scores += [batch_pos_scores[j]]
                except:
                    offset = int(j % len(batch_pos))
                    texts_q += [batch_pos[offset]]
                    clf_scores += [batch_pos_scores[offset]]

            for j in range(self.m_negatives):
                texts_p += [p]
                clf_labels += [0]
                try:
                    texts_q += [batch_neg[j]]
                    clf_scores += [batch_neg_scores[j]]
                except:
                    offset = int(j % len(batch_neg)) 
                    texts_q += [batch_neg[offset]]
                    clf_scores += [batch_neg_scores[offset]]

        if self.is_train:
            inputs = self.tokenizer(
                    texts_p,
                    max_length=self.max_p_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors=self.return_tensors
            )

            targets = self.tokenizer(
                    texts_q,
                    padding='max_length',
                    truncation=True,
                    return_tensors=self.return_tensors,
                    max_length=self.max_q_length
            )

            target_ids = targets['input_ids']
            target_mask = targets['attention_mask'].bool()
            target_ids = target_ids.masked_fill(~target_mask, -100)

            inputs['labels'] = target_ids
            inputs['decoder_attention_mask'] = target_mask
            inputs['clf_labels'] = torch.Tensor(clf_labels)
            inputs['clf_scores'] = torch.Tensor(clf_scores)
            inputs['batch_id'] = torch.Tensor(batch_id)

        else:
            inputs = self.tokenizer(
                    [batch['passage'] for batch in features],
                    max_length=self.max_p_length,
                    truncation=True,
                    return_tensors=self.return_tensors
            )
            inputs['passage'] = [batch['passage'] for batch in features]

            if self.is_eval:
                inputs['positive'] = [batch['positive'] for batch in features]
                inputs['negative'] = [batch['negative'] for batch in features]

        return inputs

@dataclass
class DataCollatorForMaskQG(DataCollatorBase):
    is_train: Union[bool, str] = False
    is_eval: Union[bool, str] = False
    m_negatives: int = 2
    m_positives: int = 2
    use_clf_score: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # text and id info 
        src_text = []
        tgt_text = []

        for i, batch in enumerate(features):

            p = batch['passage']
            q_texts = (batch['positive']*self.m_positives)[:self.m_positives]
            r_scores = (batch['positive_score']*self.m_positives)[:self.m_positives]

            for (r, q) in zip(r_scores, q_texts): 
                r = '%.3f'%r
                if random.uniform(0, 1) > 1: # 10 % generate relevance
                    src_text += [f"Passage: {p} Relevance: <mask> Query: {q}"]
                    tgt_text += [r]
                else:
                    src_text += [f"Passage: {p} Relevance: {r} Query: <mask>"]
                    tgt_text += [q]

            q_texts = (batch['negative']*self.m_negatives)[:self.m_negatives]
            r_scores = (batch['negative_score']*self.m_negatives)[:self.m_negatives]

            for (r, q) in zip(r_scores, q_texts): 
                r = '%.3f'%r
                if random.uniform(0, 1) > 1: # 10 % generate relevance
                    src_text += [f"Passage: {p} Relevance: <mask> Query: {q}"]
                    tgt_text += [r]
                else:
                    src_text += [f"Passage: {p} Relevance: {r} Query: <mask>"]
                    tgt_text += [q]

        if self.is_train:
            inputs = self.tokenizer(
                    src_text,
                    max_length=self.max_p_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors=self.return_tensors
            )

            targets = self.tokenizer(
                    tgt_text,
                    padding='max_length',
                    truncation=True,
                    return_tensors=self.return_tensors,
                    max_length=self.max_q_length
            )

            target_ids = targets['input_ids']
            target_mask = targets['attention_mask'].bool()
            target_ids = target_ids.masked_fill(~target_mask, -100)

            inputs['labels'] = target_ids
            inputs['decoder_attention_mask'] = target_mask

        return inputs

