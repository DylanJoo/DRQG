from transformers import Trainer, Seq2SeqTrainer
import math
import torch
import copy
from torch.nn import CrossEntropyLoss, NLLLoss, MSELoss, KLDivLoss
from torch.nn import functional as F

class TrainerBase(Trainer):

    def _verbose_prediction(
        self, 
        model, 
        input_ids, 
        attention_mask, 
        labels,
    ):
        with torch.no_grad():
            # generate the normal one
            model.eval()
            n = input_ids.size()[0]
            m = 11

            input_ids = input_ids.repeat_interleave(m, 0)
            attention_mask = attention_mask.repeat_interleave(m, 0)
            clf_scores = torch.arange(0, m, 1)/(m-1)
            clf_scores = clf_scores.repeat(n, 1)
            out = model.generate(
                    input_ids, 
                    attention_mask=attention_mask, 
                    clf_scores=clf_scores,
                    clf_labels=clf_labels,
                    return_dict_in_generate=True,
                    output_scores=True,
                    num_beams=1
            )
            temp = out.sequences
            labels_reformulate = [l for l in labels[0] if l != -100]
            print("D2Q+ *", model.tokenizer.decode(labels_reformulate, skip_special_tokens=True))
            for i in range(model.n_samples):
                print(f"D2Q ({i:<3}):", 
                        model.tokenizer.decode(
                            temp[i*n], skip_special_tokens=True
                        )
                )
            model.train()

class TrainerForVQG(TrainerBase):

    def compute_loss(self, model, inputs, return_outputs=False):

        training_steps = copy.deepcopy(self.state.global_step)
        outputs = model(**inputs, steps=training_steps)

        # inputs
        # labels = inputs.get("labels").to(model.device)
        clf_labels = inputs.get("clf_labels") #hard label
        clf_scores = inputs.get("clf_scores") #soft label

        # outputs
        lm_logits = outputs.get("logits")
        clf_logits = outputs.get("clf_logits")

        ## (1) CE Loss (but separate positive/negative)
        loss_gen_pos, loss_gen_neg = 0, 0
        loss_fct = CrossEntropyLoss(reduction='none')
        select_pos = (clf_labels==1)
        loss_gen_pos = loss_fct(
                lm_logits[select_pos].view(-1, model.config.vocab_size), 
                labels[selec_post].view(-1)
        ).mean()
        select_neg = (clf_labels<1)
        loss_gen_neg = loss_fct(
                lm_logits[select_neg].view(-1, model.config.vocab_size), 
                labels[select_neg].view(-1)
        ).mean()
        loss_gen = (loss_gen_pos+loss_gen_neg) / 2

        ## (2) KL loss (relevance)
        loss_clf = 0
        loss_fct = KLDivLoss(reduction='sum')
        clf_logp = F.log_softmax(clf_logits.view(-1, 2), -1) # BL 2
        target_p = torch.cat([(1-clf_scores).view(-1, 1), clf_scores.view(-1, 1)], -1)
        loss_rel = loss_fct(clf_logp, target_p)
        loss_rel = loss_rel / labels.size(0)

        ## (4) KL loss (reparam)
        loss_reparam = outputs.get('reparam_loss')

        loss = loss_gen + loss_reparam + loss_rel

        # [NOTE] add evaluation for monitoring
        if training_steps % 50 == 0:
            print(f"\nNLL: (pos) {loss_gen_pos} (neg) {loss_gen_neg} \
                    \nKL: (reparam) {loss_reparam} (rel) {loss_rel}")

            if select_pos.sum() == 0:
                select_pos[0] = True
            inputs_for_eval = {
                    "input_ids": inputs['input_ids'][select_pos],
                    "attention_mask": inputs['attention_mask'][select_pos],
                    "labels": inputs['labels'][select_pos]
            }
            self._verbose_prediction(model, **inputs_for_eval)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss
    
