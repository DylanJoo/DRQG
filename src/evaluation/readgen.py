import itertools
import torch
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import AutoConfig, AutoTokenizer
from .generator import SoftPromptFlanT5

class READGen:
    def __init__(
        self,
        model_path: str = None, 
        tokenizer_name: str = None, 
        relevance_scores: str = None, 
        num_relevance_scores: int = 10, 
        output_jsonl: Optional[str] = None,
        device: str = 'cpu',
        **kwargs
    ):
        ## readgen config
        self.num_relevance_scores = num_relevance_scores
        self.output_path = output_jsonl
        self.device = device

        ## model config and tokenizers
        self.model = SoftPromptFlanT5.from_pretrained(
                model_path,
                config=AutoConfig.from_pretrained(model_path),
                num_instruction_prompt_idx=\
                        kwargs.get('num_instruction_prompt_idx', 13),
                num_relevant_prompt_idx=\
                        kwargs.get('num_relevant_prompt_idx', 0)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.prompt_length = sum(self.model.prompt_length)

        if relevance_scores is None:
            relevance_scores = list(range(0,101, num_relevance_scores))
        self.relevance_scores = torch.Tensor(
                [s*0.01 for s in relevance_scores]).to(device)

    def _expand(self, mask):
        additional_mask = torch.ones(
                (mask.size(0), self.prompt_length), device=self.device
        )
        return torch.cat([additional_mask, mask], -1)

    def batch_generate(self, text_inputs, prefix=None, **kwargs):
        # prepare inputs 
        M = len(text_inputs) # batch_size
        N = len(self.relevance_scores) # N

        # tokenization
        if prefix is not None:
            scaled_relevance_scores = [round(r*100) for r in self.relevance_scores.detach().cpu().numpy()]
            text_inputs_with_rel = list(
                    itertools.product(text_inputs, scaled_relevance_scores)
            )
            text_inputs_with_rel = [(s, t) for (t, s) in text_inputs_with_rel] # reverse
            inputs = self.tokenizer(
                    [prefix.format(s, t) for (s, t) in text_inputs_with_rel], 
                    max_length=kwargs.pop('max_length', 512),
                    truncation=kwargs.pop('truncation', True),
                    padding=kwargs.pop('padding', True),
                    return_tensors='pt'
            ).to(self.device)
            input_ids = inputs['input_ids']
            attn_mask = inputs['attention_mask']
            rel_scores = self.relevance_scores.repeat(M)
            attn_mask = self._expand(attn_mask)
        else:
            inputs = self.tokenizer(
                    text_inputs, 
                    max_length=kwargs.pop('max_length', 512),
                    truncation=kwargs.pop('truncation', True),
                    padding=kwargs.pop('padding', True),
                    return_tensors='pt'
            ).to(self.device)
            input_ids = inputs['input_ids'].repeat_interleave(N, 0)
            attn_mask = inputs['attention_mask'].repeat_interleave(N, 0)
            rel_scores = self.relevance_scores.repeat(M)
            attn_mask = self._expand(attn_mask)

        # generate
        output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                rel_scores=rel_scores,
                **kwargs
        )

        # outputs (if any)
        output_texts = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
        )
        ## return a list of list, each list contains N predicted query
        return [output_texts[i:i+N] for i in range(0, N*M, N)]
