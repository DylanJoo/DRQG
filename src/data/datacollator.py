""" The datacollator for pcentric dataset.
"""
import torch
import random
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

@dataclass
class DataCollatorForCtrlQG(DataCollatorBase):
    is_train: Union[bool, str] = False
    is_eval: Union[bool, str] = False
    m_negatives: int = 2
    m_positives: int = 2
    use_clf_score: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # text and id info 
        texts_p = []
        texts_q = []
        labels = []
        scores = []
        batch_id = []

        for i, batch in enumerate(features):
            p = batch['passage']

            # positive
            q1_batch = batch['positive']
            score1_batch = batch['positive_score']
            for j in range(self.m_positives):
                texts_p += [p]
                labels += [1]
                try:
                    texts_q += [q1_batch[j]]
                    scores += [score1_batch[j]]
                except: # sometimes #available is less than #specififed 
                    offset = int(j % len(q1_batch))
                    texts_q += [q1_batch[offset]]
                    scores += [score1_batch[offset]]

            # negative
            q0_batch = batch['negative'][::-1]
            score0_batch = batch['negative_score'][::-1]
            for j in range(self.m_negatives):
                texts_p += [p]
                labels += [0]
                try:
                    texts_q += [q0_batch[j]]
                    scores += [score0_batch[j]]
                except:
                    offset = int(j % len(q0_batch)) 
                    texts_q += [q0_batch[offset]]
                    scores += [score0_batch[offset]]

        inputs = self.tokenizer(
                texts_p,
                max_length=self.max_p_length,
                truncation=True,
                padding='max_length',
                return_tensors=self.return_tensors
        )

        targets = self.tokenizer(
                texts_q,
                max_length=self.max_q_length,
                truncation=True,
                padding='max_length',
                return_tensors=self.return_tensors
        )

        target_ids = targets['input_ids']
        target_mask = targets['attention_mask'].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        inputs['labels'] = target_ids
        inputs['decoder_attention_mask'] = target_mask
        inputs['rel_labels'] = torch.Tensor(labels)
        inputs['rel_scores'] = torch.Tensor(scores)
        return inputs

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
