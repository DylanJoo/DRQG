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
    is_train: Union[bool, str] = False
    # spec
    p_centric: Optional[bool] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # text and id info 
        if self.p_centric:
            texts_p = [batch['passage'] for batch in features]
            texts_pq = [batch['positive'] for batch in features]
            texts_nq = [batch['negative'] for batch in features]

        # positive 
        inputs = self.tokenizer(
                [f"<extra_id_10> {p}" for p in texts_p] * 2 ,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors=self.return_tensors
        )

        if self.is_train:
            targets = self.tokenizer(
                    texts_pq+texts_nq,
                    padding=True,
                    return_tensors=self.return_tensors
            ).input_ids
            inputs['labels'] = targets

        return inputs
