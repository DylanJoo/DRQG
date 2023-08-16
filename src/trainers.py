from transformers import Trainer, Seq2SeqTrainer
import math
import torch
import copy
from torch.nn import CrossEntropyLoss, NLLLoss, MSELoss, KLDivLoss
from torch.nn import functional as F
from models.loss import gen_mle_loss

class TrainerBase(Seq2SeqTrainer):

    def set_tokenizer(self, tokenizer=None):
        self.tokenizer = tokenizer

    def set_prefix(self, prefix=None):
        self.prefix = prefix

    def compute_loss(self, model, inputs):
        """
        param: text_inputs: the raw inputs of passages.
        """
        passage = inputs.pop('passage')
        training_steps = copy.deepcopy(self.state.global_step)
        loss = super().compute_loss(model, inputs, return_outputs)

        if training_steps % 50 == 0:
            print(f"\nNLL: {loss}")
            self._verbose_prediction(model, passage, 10)
        return loss

    def _verbose_prediction(self, model, passage, num_pred=10):
        """
        param: passage: one passage for prediction.
        param: num_pred: number of interpolated output (range from 0-100)
        """
        # construct relevance score conditions
        scores = list(range(0, 101, 101//(num_pred-1)))
        if self.prefix:
            text_inputs = [self.prefix.format(s, passage) for s in scores]
        else:
            text_inputs = [passage for s in scores]
            scores = torch.tensor(scores) / 100

        inputs = self.tokenizer(
                text_inputs, 
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors='pt'
        )
        for k in inputs:
            inputs[k] = inputs[k].to(model.device)

        model.eval()
        with torch.no_grad():
            outputs = model.generate(**inputs, num_beams=1)

            print('============')
            print("Passage", passage)
            print('============')
            for i, s in enumerate(scores):
                print(f"({i:<3}) >>", self.tokenizer.decode(outputs[i], skip_special_tokens=True))
        model.train()

class TrainerForQG(TrainerBase):

    def compute_loss(self, model, inputs, return_outputs=False):
        passage = inputs.pop('passage')
        training_steps = copy.deepcopy(self.state.global_step)
        outputs = model(**inputs, steps=training_steps)

        # inputs
        labels = inputs.get("labels").to(model.device)

        # outputs
        lm_logits = outputs.get("logits")

        ## (1) text generation loss
        loss_gen = gen_mle_loss(
                lm_logits, 
                labels, 
                inputs.get('rel_labels', None),
                model.config.vocab_size
        )
        loss_gen_pos = loss_gen.get('pos', 0)
        loss_gen_neg = loss_gen.get('neg', 0)

        loss_gen = (loss_gen_pos+loss_gen_neg) / 2
        loss = loss_gen

        if training_steps % 50 == 0:
            print(f"\nNLL: {loss}")
            self._verbose_prediction(model, passage, 10)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss
    
