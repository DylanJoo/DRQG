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

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # text and id info 
        texts_p = [batch['passage'] for batch in features]
        texts_pq = [batch['positive'] for batch in features]
        texts_nq = [batch['negative'] for batch in features]

        if self.is_train:
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
            inputs['labels'] = targets

        elif self.is_eval:
            inputs = self.tokenizer(
                    [f"<extra_id_10> {p}" for p in texts_p],
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors=self.return_tensors
            )
            inputs['passage'] = texts_p
            inputs['positive'] = texts_pq
            inputs['negative'] = texts_nq
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

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # text and id info 
        texts_p = [batch['passage'] for batch in features]
        texts_pq = [batch['positive'] for batch in features]
        texts_nq = [batch['negative'] for batch in features]

        if self.is_train:
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
            inputs['labels'] = targets

        elif self.is_eval:
            inputs = self.tokenizer(
                    [f"positive question generation: passage: {p}" for p in texts_p] + \
                    [f"negative question generation: passage: {p}" for p in texts_p],
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors=self.return_tensors
            )
            inputs['passage'] = texts_p
            inputs['positive'] = texts_pq
            inputs['negative'] = texts_nq
        return inputs
