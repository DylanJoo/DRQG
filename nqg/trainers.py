from transformers import Trainer, Seq2SeqTrainer
import math
import torch
import copy
from torch.nn import CrossEntropyLoss, NLLLoss, MSELoss, KLDivLoss
from torch.nn import functional as F

class TrainerBase(Trainer):

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
        with torch.no_grad():
            # generate the normal one
            model.eval()
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
            labels_reformulate = [l for l in labels[0] if l != -100]
            print("D2Q+ *", model.tokenizer.decode(labels_reformulate, skip_special_tokens=True))
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

        ## (1) CE Loss (same as built-in but separate positive/negative)
        ### masked labels
        rand = torch.rand(labels.shape, device=labels.device) 
        masked = (rand > 0.7).to(labels.device)
        masked = masked & (clf_labels==0).view(-1, 1).repeat(1, labels.size(1))
        labels = labels.masked_fill(masked, -100)

        w = clf_scores.exp().view(-1, 1).expand(labels.shape)
        loss_gen_pos, loss_gen_neg = 0, 0
        loss_fct = CrossEntropyLoss(reduction='none')
        selected = (clf_labels<1)
        loss_gen_neg = loss_fct(
                lm_logits[selected].view(-1, model.config.vocab_size), 
                labels[selected].view(-1)
        )
        loss_gen_neg = loss_gen_neg.mean()

        loss_fct = CrossEntropyLoss(reduction='none')
        selected = (clf_labels==1)
        loss_gen_pos = loss_fct(
                lm_logits[selected].view(-1, model.config.vocab_size), 
                labels[selected].view(-1)
        )
        loss_gen_pos = loss_gen_pos.mean()

        loss_gen = (loss_gen_pos+loss_gen_neg) / 2

        ## (2) KL loss (relevance)
        loss_rel = 0
        loss_fct = KLDivLoss(reduction='sum')
        logp = F.log_softmax(clf_logits.view(-1, 2), -1) # BL 2
        target = torch.cat([(1-clf_scores).view(-1, 1), clf_scores.view(-1, 1)], -1)
        loss_rel = loss_fct(logp, target)
        loss_rel = loss_rel / labels.size(0)

        ## (4) KL loss (reparam)
        loss_reparam = outputs.get('reparam_loss')

        loss = loss_gen+loss_reparam+loss_rel
        # loss = loss_reparam+loss_rel

        # [NOTE] add evaluation for monitoring
        if training_steps % 50 == 0:
            print(f"\nNLL: (pos) {loss_gen_pos} (neg) {loss_gen_neg} \
                    \nKL: (reparam) {loss_reparam} (rel) {loss_rel}")

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
    
