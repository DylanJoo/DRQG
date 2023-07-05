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
                    "input_ids": inputs['input_ids'],
                    "attention_mask": inputs['attention_mask'],
                    "labels": inputs['labels'],
                    "m": 1
            }
            self._verbose_prediction(model, **inputs_for_eval)

        return outputs

    def _verbose_prediction(
        self, 
        model, 
        input_ids, 
        attention_mask, 
        labels,
        m
    ):
        model.eval()
        with torch.no_grad():
            # generate the normal one
            n = input_ids.size()[0]

            if m>1:
                input_ids = input_ids.repeat_interleave(m, 0)
                attention_mask = attention_mask.repeat_interleave(m, 0)
                clf_scores = torch.arange(0, m, 1)/(m-1)
                clf_scores = clf_scores.repeat(n)
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
            for i in range(4):
                labels_reformulate = [l for l in labels[i] if l != -100]
                print("D2Q+ *", model.tokenizer.decode(labels_reformulate, skip_special_tokens=True))

            print('===')
            for i in range(m):
                print(f"D2Q ({i:<3}):", 
                        model.tokenizer.decode(temp[i], skip_special_tokens=True)
                )

        model.train()

class TrainerForQG(TrainerBase):

    def compute_loss(self, model, inputs, return_outputs=False):

        training_steps = copy.deepcopy(self.state.global_step)
        outputs = model(**inputs, steps=training_steps)

        # inputs
        labels = inputs.get("labels").to(model.device)
        clf_labels = inputs.get("clf_labels") #hard label
        clf_scores = inputs.get("clf_scores") #soft label

        # outputs
        lm_logits = outputs.get("logits")
        clf_logits = outputs.get("clf_logits")

        ## (1) text generation loss
        loss_gen = gen_mle_loss(
                lm_logits, labels, clf_labels, 
                model.config.vocab_size
        )
        loss_gen_pos = loss_gen.get('pos', 0)
        loss_gen_neg = loss_gen.get('neg', 0)

        ### (1.a) text generation loss with gumbel softmax
        # loss_gen_gumbel = gen_mle_gumbel_loss(
        #         lm_logits, labels, clf_labels, 
        #         model.config.vocab_size, training_steps
        # )
        # loss_gen_pos = loss_gen_gumbel.get('pos', 0)
        # loss_gen_neg = loss_gen_gumbel.get('neg', 0)

        loss_gen = (loss_gen_pos+loss_gen_neg) / 2

        ## (2) query logits contrastive loss
        loss_cont = outputs.get("cont_loss", 0)

        ## (3) relevance prediction soft loss
        loss_rel = 0
        loss_rel = ql_kl_loss(clf_logits, clf_scores)

        ## (4) KL loss (reparam)
        loss_reparam = outputs.get('reparam_loss', 0)

        loss = loss_gen
        if training_steps is not None:
            loss += loss_cont

        # [NOTE] add evaluation for monitoring
        if training_steps % 50 == 0:
            selected = (clf_labels==1)
            print(f"\nNLL: (pos) {loss_gen_pos} (neg) {loss_gen_neg} \
                    \nKL: (reparam) {loss_reparam} (rel) {loss_rel} \
                    \nCE: (condition) {loss_cont}")

            inputs_for_eval = {
                    "input_ids": inputs['input_ids'][selected],
                    "attention_mask": inputs['attention_mask'][selected],
                    "labels": inputs['labels'][selected],
                    "m": 11
            }
            self._verbose_prediction(model, **inputs_for_eval)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss
    
