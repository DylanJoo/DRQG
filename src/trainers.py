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

    def compute_loss(self, model, inputs):
        """
        param: text_inputs: the raw inputs of passages.
        """
        passage = inputs.pop('passage', "")
        training_steps = copy.deepcopy(self.state.global_step)
        loss = super().compute_loss(model, inputs, return_outputs)

        if training_steps % 50 == 0:
            print(f"\nNLL: {loss}")
            self._verbose_prediction(model, passage)
        return loss

    def _verbose_prediction(self, model, passage):
        """
        param: model: a generator or a seq2seq model.
        param: passage: one passage for prediction.
        """
        # construct relevance score conditions
        features = [{'passage': passage}]
        inputs, _ = self.data_collator(features, is_eval=True)
        inputs = inputs.to(model.device)

        model.eval()
        with torch.no_grad():
            outputs = model.generate(**inputs, num_beams=1)
                    # rel_scores=torch.Tensor(self.data_collator.scores),
            print('============\nPassage: ', passage, '\n============')
            for i, s in enumerate(self.data_collator.scores):
                print(f"({i:<3}) >>", self.tokenizer.decode(outputs[i], skip_special_tokens=True))
        model.train()

class TrainerForQG(TrainerBase):

    def compute_loss(self, model, inputs, return_outputs=False):
        passage = inputs.pop('passage')
        rel_labels = inputs.pop('rel_labels')
        training_steps = copy.deepcopy(self.state.global_step)
        outputs = model(**inputs, steps=training_steps)

        # outputs
        lm_logits = outputs.get("logits")
        n = lm_logits.shape[0] // 2

        # inputs
        labels = inputs.get("labels").to(lm_logits.device)

        ## (1) text generation loss
        loss_gen = gen_mle_loss(
                lm_logits, 
                labels, 
                rel_labels,
                model.config.vocab_size
        )
        loss_gen_pos, loss_gen_neg = loss_gen['pos'], loss_gen['neg']

        loss = (loss_gen_pos + loss_gen_neg) / 2

        if training_steps % 50 == 0:
            print(f"\nNLL: {loss} = {loss_gen_pos} + {loss_gen_neg}")
            self._verbose_prediction(model, passage)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss
    
class TrainerForRelQG(TrainerForQG):

    def _verbose_prediction(self, model, passage):
        """
        param: model: a generator or a seq2seq model.
        param: passage: one passage for prediction.
        """
        # construct relevance score conditions
        features = [{'passage': passage}]
        inputs, _ = self.data_collator(features, is_eval=True)
        inputs = inputs.to(model.device)

        rel_scores = torch.Tensor(self.data_collator.scores).to(model.device)
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                    **inputs, 
                    rel_scores=rel_scores,
                    num_beams=1
            )
            print('============\nPassage: ', passage, '\n============')
            for i, s in enumerate(self.data_collator.scores):
                print(f"({i:<3}) >>", self.tokenizer.decode(outputs[i], skip_special_tokens=True))
        model.train()
