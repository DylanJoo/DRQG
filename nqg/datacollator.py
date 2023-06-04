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
                texts_q = texts_q1
            if self.relevant_included:
                texts_q = texts_q0

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
class DataCollatorForPQG(DataCollatorBase):
    is_train: Union[bool, str] = False
    is_eval: Union[bool, str] = False
    prefix: str = "positive question generation: passage: "
    negative_prefix: str = "negative question generation: passage: "

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # text and id info 
        texts_p = [batch['passage'] for batch in features]

        if self.is_train:
            texts_pq = [batch['positive'] for batch in features]
            texts_nq = [batch['negative'] for batch in features]
            inputs = self.tokenizer(
                    [f"{self.prefix}{p}" for p in texts_p] + \
                    [f"{self.negative_prefix} {p}" for p in texts_p],
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors=self.return_tensors
            )
            targets = self.tokenizer(
                    texts_pq+texts_nq,
                    padding=True,
                    return_tensors=self.return_tensors
            ).input_ids
            targets[targets == self.tokenizer.pad_token_id] = -100
            inputs['labels'] = targets
            return inputs

        else:
            inputs1 = self.tokenizer(
                    [f"{self.prefix}{p}" \
                            for p in texts_p],
                    max_length=self.max_p_length,
                    truncation=True,
                    padding=True,
                    return_tensors=self.return_tensors
            )
            inputs0 = self.tokenizer(
                    [f"{self.negative_prefix}{p}" \
                            for p in texts_p],
                    max_length=self.max_p_length,
                    truncation=True,
                    padding=True,
                    return_tensors=self.return_tensors
            )
            inputs = {'passage': texts_p}

            if self.is_eval:
                inputs['positive'] = [batch['positive'] for batch in features]
                inputs['negative'] = [batch['negative'] for batch in features]
            return inputs, inputs1, inputs0


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

        for i, batch in enumerate(features):

            for i, batch_pos in enumerate(batch['positive'][:self.m_positives]):
                texts_p += [batch['passage']]
                texts_q += [batch_pos]
                clf_labels += [1]
                if self.use_clf_score:
                    clf_scores += [batch['positive_score'][i]]

            for i, batch_neg in enumerate(batch['negative'][::-1][:self.m_negatives]):
                texts_p += [batch['passage']]
                texts_q += [batch_neg]
                clf_labels += [0]
                if self.use_clf_score:
                    clf_scores += [batch['negative_score'][::-1][i]]

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
class DataCollatorForVQG2(DataCollatorBase):
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

        for i, batch in enumerate(features):

            for i, batch_pos in enumerate(batch['positive'][:self.m_positives]):
                texts_p += [f"{batch['passage']} | {batch_pos}"]
                texts_q += [batch_pos]
                clf_labels += [1]
                if self.use_clf_score:
                    clf_scores += [batch['positive_score'][i]]

            for i, batch_neg in enumerate(batch['negative'][::-1][:self.m_negatives]):
                texts_p += [f"{batch['passage']} | {texts_q[0]}"]
                texts_q += [batch_neg]
                clf_labels += [0]
                if self.use_clf_score:
                    clf_scores += [batch['negative_score'][::-1][i]]

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
            inputs['clf_labels'] = torch.LongTensor(clf_labels)
            inputs['clf_scores'] = torch.Tensor(clf_scores)

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
class DataCollatorForSQUAD(DataCollatorBase):
    is_train: Union[bool, str] = False
    is_eval: Union[bool, str] = False
    m_negatives: int = 2

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # text and id info 
        texts_p = []
        texts_q = []
        clf_labels = []

        for i, batch in enumerate(features):
            texts_q += batch['positive'][:self.m_negatives]
            n = len(batch['positive'][:self.m_negatives])
            texts_p += [batch['passage']] * n
            clf_labels += [1]+[0]*(n-1)

        if self.is_train:
            inputs = self.tokenizer(
                    texts_p,
                    max_length=self.max_p_length,
                    truncation=True,
                    padding=True,
                    return_tensors=self.return_tensors
            )

            targets = self.tokenizer(
                    texts_q,
                    padding=True,
                    return_tensors=self.return_tensors,
                    max_length=self.max_q_length
            )

            target_ids = targets['input_ids']
            target_mask = targets['attention_mask'].bool()
            target_ids = target_ids.masked_fill(~target_mask, -100)

            inputs['labels'] = target_ids
            inputs['decoder_attention_mask'] = target_mask
            inputs['clf_labels'] = torch.LongTensor(clf_labels)

        else:
            pass
        return inputs

