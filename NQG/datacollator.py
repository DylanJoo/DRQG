import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

@dataclass
class DataCollatorForT5VAE:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True
    istrain: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # text and id info 
        texts_q = [batch['query'] for batch in features]
        texts_pp = [batch['positive'] for batch in features]
        texts_pn = [batch['negative'] for batch in features]

        # positive 
        inputs = self.tokenizer(
                [f"<extra_id_10> {p}" for p in texts_pp] + [f"{p}" for p in texts_pn], 
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors=self.return_tensors
        )

        if self.istrain:
            targets = self.tokenizer(
                    texts_q + texts_q,
                    padding=True,
                    return_tensors=self.return_tensors
            ).input_ids
            inputs['labels'] = targets

        return inputs



@dataclass
class DataCollatorForBartVAE:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True
    istrain: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # text and id info 
        texts_q = [batch['query'] for batch in features]
        texts_pp = [batch['positive'] for batch in features]
        texts_pn = [batch['negative'] for batch in features]

        # positive 
        inputs = self.tokenizer(
                [f"document: {p}" for p in texts_pp + texts_pn], 
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors=self.return_tensors
        )

        if self.istrain:
            targets = self.tokenizer(
                    texts_q + texts_q,
                    padding=True,
                    return_tensors=self.return_tensors
            ).input_ids
            inputs['labels'] = targets

        return inputs


