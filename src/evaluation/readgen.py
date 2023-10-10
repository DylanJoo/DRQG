import torch
from transformers import AutoConfig, AutoTokenizer
from generator import SoftPromptFlanT5

class READGen:
    def __init__(
        self,
        model_path: str = None, 
        tokenizer_name: str = None, 
        relevance_scores: str = None, 
        num_relevance_scores: int = 10, 
        output_jsonl: Optional[str] = None,
        **kwargs
    ):
        ## readgen config
        self.num_relevance_scores = num_relevance_scores
        self.output_path = output_jsonl

        ## model config and tokenizers
        self.model = SoftPromptFlanT5.from_pretrained(
                model_path,
                AutoConfig.from_pretrained(model_path),
                kwargs.get('num_instruction_prompt_idx', 13),
                kwargs.get('num_relevance_prompt_idx', 1)
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

    def batch_generate(self, text_inputs, **kwargs):
        # prepare inputs 
        M = len(text_inputs)
        rel_scores = self.relevance_scores.repeat(M)

        # tokenization
        inputs = self.tokenizer(
                text_inputs, 
                max_length=kwargs.pop('max_length', 512),
                truncation=kwargs.pop('truncation', True),
                padding=kwargs.pop('padding', True)
                return_tensors=True
        ).to(self.device)

        N = len(rel_scores)
        input_ids = inputs['input_ids'].repeat_interleave(N, 1)
        attn_mask = inputs['attention_mask'].repeat_interleave(N, 1)

        # generate
        output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask._expand(attn_mask),
                rel_scores=rel_scores,
                **kwargs
        )

        # outputs (if any)
        output_texts = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
        )
        ## return a list of list, each list contains N predicted query
        return [output_texts[i:i+N] for i in range(0, N*M, N)]
