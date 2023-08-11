from transformers import Trainer, Seq2SeqTrainer
import math
import torch
import copy
from torch.nn import CrossEntropyLoss, NLLLoss, MSELoss, KLDivLoss
from torch.nn import functional as F
from models.loss import (
        gen_mle_loss,
        ql_kl_loss,
        gen_mle_gumbel_loss
)

class TrainerBase(Seq2SeqTrainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        training_steps = copy.deepcopy(self.state.global_step)
        outputs = super().compute_loss(model, inputs, return_outputs)

        if training_steps % 50 == 0:
            print(f"\nNLL: {outputs}")
            inputs_for_eval = {
                    "input_ids": inputs['input_ids'][0],
                    "attention_mask": inputs['attention_mask'][0],
                    "labels": inputs['labels'][0],
                    "m": 1
            }
            self._verbose_prediction(model, **inputs_for_eval)

        return outputs

    def _verbose_prediction(self, model, input_ids, attention_mask, labels, m):
        model.eval()
        with torch.no_grad():
            # generate the normal one
            n = input_ids.size()[0]

            if m>1:
                input_ids = input_ids.repeat_interleave(m, 0)
                attention_mask = attention_mask.repeat_interleave(m, 0)
                scores = torch.arange(0, m, 1)/(m-1)
                scores = scores.repeat(n)
                out = model.generate(
                        input_ids, 
                        attention_mask=attention_mask, 
                        clf_scores=clf_scores,
                        clf_labels=clf_scores,
                        return_dict_in_generate=True,
                        output_scores=True,
                        num_beams=1
                )
            else:
                out = model.generate(
                        input_ids, 
                        attention_mask=attention_mask, 
                        return_dict_in_generate=True,
                        output_scores=True,
                        num_beams=1
                )
            temp = out.sequences
            m = len(temp) if m == 1 else m

            print('===')
            labels_reformulate = [l for l in labels[0] if l != -100]
            print("Truth query+:", model.tokenizer.decode(labels_reformulate, skip_special_tokens=True))

            print('===')
            for i in range(m):
                print(f"Predicted query ({i:<3}):", 
                        model.tokenizer.decode(temp[i], skip_special_tokens=True)
                )

        model.train()

class TrainerForQG(TrainerBase):

    def compute_loss(self, model, inputs, return_outputs=False):

        training_steps = copy.deepcopy(self.state.global_step)
        outputs = model(**inputs, steps=training_steps)

        # inputs
        labels = inputs.get("labels").to(model.device)
        rel_labels = inputs.get("rel_labels") #hard label
        rel_scores = inputs.get("rel_scores") #soft label

        # outputs
        lm_logits = outputs.get("logits")

        ## (1) text generation loss
        loss_gen = gen_mle_loss(
                lm_logits, labels, clf_labels, 
                model.config.vocab_size
        )
        loss_gen_pos = loss_gen.get('pos', 0)
        loss_gen_neg = loss_gen.get('neg', 0)

        loss_gen = (loss_gen_pos+loss_gen_neg) / 2
        loss = loss_gen

        # [NOTE] add evaluation for monitoring
        if training_steps % 50 == 0:
            print(f"\nNLL: (pos) {loss_gen_pos} (neg) {loss_gen_neg}")

            inputs_for_eval = {
                    "input_ids": inputs['input_ids'][0],
                    "attention_mask": inputs['attention_mask'][0],
                    "labels": inputs['labels'][0],
                    "m": 11
            }
            self._verbose_prediction(model, **inputs_for_eval)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss
    
