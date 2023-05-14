"""
The datacollator for pcentric dataset.
"""
import random
import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

@dataclass
class DataCollatorForT5VQG:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True
    is_train: Union[bool, str] = False
    is_eval: Union[bool, str] = False
    # spec
    m_samples_per_example: int = 1

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # text and id info 
        texts_p = [batch['passage'] for batch in features]

        if self.is_train:
            texts_pq = [batch['positive'] for batch in features]
            texts_nq = [batch['negative'] for batch in features]
            """
            When training, a batch contains two passaegs, this is for 
            optimizing NQG and PQF(doc2query) in the same batch with
            discrepancy loss of same passage but different target.
            """
            inputs = self.tokenizer(
                    [f"<extra_id_10> {p}" for p in texts_p] * 2 ,
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

        else:
            """
            When inferencing, a batch contains only one passaegs. 
            Each passages is the to-be-predicted instance.
            """
            inputs = self.tokenizer(
                    [f"<extra_id_10> {p}" for p in texts_p],
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors=self.return_tensors
            )
            inputs['passage'] = texts_p

            if self.is_eval:
                inputs['positive'] = [batch['positive'] for batch in features]
                inputs['negative'] = [batch['negative'] for batch in features]
        return inputs

@dataclass
class DataCollatorForT5PQG: # prompt-T5 generation
    tokenizer: Union[PreTrainedTokenizerBase] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True
    is_train: Union[bool, str] = False
    is_eval: Union[bool, str] = False
    # spec
    m_samples_per_example: int = 1

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # text and id info 
        texts_p = [batch['passage'] for batch in features]

        if self.is_train:
            texts_pq = [batch['positive'] for batch in features]
            texts_nq = [batch['negative'] for batch in features]
            inputs = self.tokenizer(
                    [f"positive question generation: passage: {p}" for p in texts_p] + \
                    [f"negative question generation: passage: {p}" for p in texts_p],
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
                    [f"positive question generation: passage: {p}" \
                            for p in texts_p],
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors=self.return_tensors
            )
            inputs0 = self.tokenizer(
                    [f"negative question generation: passage: {p}" \
                            for p in texts_p],
                    max_length=self.max_length,
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
class DataCollatorForVQGSPT:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True
    is_train: Union[bool, str] = False
    is_eval: Union[bool, str] = False
    # spec
    m_samples_per_example: int = 1

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # text and id info 
        texts_p1 = [batch['passage'] for batch in features]
        texts_p0 = [batch['passage'] for batch in features]

        if self.is_train:
            texts_pq = [batch['positive'] for batch in features]
            texts_nq = [batch['negative'] for batch in features]

            inputs = self.tokenizer(
                    texts_p1 + texts_p0,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors=self.return_tensors
            )

            targets = self.tokenizer(
                    [p for p in texts_pq+texts_nq],
                    padding=True,
                    return_tensors=self.return_tensors,
            )

            target_ids = targets['input_ids']
            target_mask = targets['attention_mask'].bool()
            target_ids = target_ids.masked_fill(~target_mask, -100)

            inputs['labels'] = target_ids
            inputs['decoder_attention_mask'] = target_mask

        else:
            inputs = self.tokenizer(
                    [p for p in texts_p],
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors=self.return_tensors
            )
            inputs['passage'] = texts_p

            if self.is_eval:
                inputs['positive'] = [batch['positive'] for batch in features]
                inputs['negative'] = [batch['negative'] for batch in features]
        return inputs

@dataclass
class DataCollatorForVQGDEV:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True
    is_train: Union[bool, str] = False
    is_eval: Union[bool, str] = False
    # spec
    m_samples_per_example: int = 2

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # text and id info 
        texts_p = []
        texts_pq = []
        texts_nq = []

        """
        Preparing (a) passage; (b) positive/ negative query 

        # p: [(b1p * m; b2p * m .. bnp * m)] * 2
        # q+: (b1q+1, b1q+2, ...b1q+m; b2q+1, b2q+2, ...b2q+m; bnq+1 ...)
        # q-: (b1q-1, b1q-2, ...b1q-m; b2q-1, b2q-2, ...b2q-m; bnq-1 ...)
        """
        for batch in features:
            texts_p += [batch['passage']] * self.m_samples_per_example

        for batch in features:
            texts_pq += batch['positive'][:self.m_samples_per_example]
            texts_nq += batch['negative'][:self.m_samples_per_example]

        if self.is_train:
            inputs = self.tokenizer(
                    texts_p + texts_p,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors=self.return_tensors
            )

            targets = self.tokenizer(
                    texts_pq + texts_nq,
                    padding=True,
                    return_tensors=self.return_tensors
            )

            target_ids = targets['input_ids']
            target_mask = targets['attention_mask'].bool()
            target_ids = target_ids.masked_fill(~target_mask, -100)

            inputs['labels'] = target_ids
            inputs['decoder_attention_mask'] = target_mask

        else:
            inputs = self.tokenizer(
                    [p for p in texts_p],
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors=self.return_tensors
            )
            inputs['passage'] = texts_p

            if self.is_eval:
                inputs['positive'] = [batch['positive'] for batch in features]
                inputs['negative'] = [batch['negative'] for batch in features]
        return inputs
