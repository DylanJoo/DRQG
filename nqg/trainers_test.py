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
            n = input_ids.size(0)
            m = 11

            # replicate n_samplet times
            input_ids = input_ids.repeat_interleave(m, 0)
            attention_mask = attention_mask.repeat_interleave(m, 0)
            clf_scores = torch.arange(0, m, 1)/(m-1)
            clf_scores = clf_scores.repeat(n, 1)
            out = model.generate(
                    input_ids, 
                    attention_mask=attention_mask, 
                    clf_scores=clf_scores,
                    clf_labels=clf_scores,
                    return_dict_in_generate=True,
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

class TrainerForCVQG(TrainerBase):

    def compute_loss(self, model, inputs, return_outputs=False):

        # [NOTE] `label_smoother` was tooked out in this trainer. 
        # [NOTE] add training steps info 
        training_steps = copy.deepcopy(self.state.global_step)
        outputs = model(**inputs, steps=training_steps)

        ## take out outputs
        # loss_gen = outputs.get("loss")
        lm_logits = outputs.get("logits")
        clf_logits = outputs.get("clf_logits")

        # Take out inputs
        labels = inputs.get("labels").to(model.device)
        clf_labels = inputs.get("clf_labels").to(model.device) #hard label
        clf_scores = inputs.get("clf_scores").to(model.device) #soft label

        # [mask] randomly selected
        # rand = torch.rand(labels.shape, device=labels.device) 
        # masked = (rand > 0.7).to(labels.device)
        # masked = masked & (clf_labels==0).view(-1, 1).repeat(1, labels.size(1))
        # labels = labels.masked_fill(masked, model.tokenizer.mask_token_id)

        ## weighted by power
        ## (1) CE Loss (but separate positive/negative)
        loss_gen_pos, loss_gen_neg = 0, 0
        length_size = labels.size(-1)
        logits = outputs.get("logits")
        selected_positive = (clf_labels==1)
        loss_fct = CrossEntropyLoss(reduction='none')
        loss_gen_pos = loss_fct(
                lm_logits[selected_positive].view(-1, model.config.vocab_size), 
                labels[selected_positive].view(-1)
        )
        selected_negative = (clf_labels!=1)
        loss_fct = CrossEntropyLoss(reduction='none')
        loss_gen_neg = loss_fct(
                lm_logits[selected_negative].view(-1, model.config.vocab_size), 
                labels[selected_negative].view(-1)
        )

        # w = clf_scores.exp().view(-1, 1).expand(labels.shape)
        # loss_gen_pos = (loss_gen_pos * w[selected_positive].view(-1)).mean()
        # loss_gen_neg = (loss_gen_neg * w[selected_negative].view(-1)).mean()

        loss_gen_pos = loss_gen_pos.mean()
        loss_gen_neg = loss_gen_neg.mean()

        ## (2) relevance kl loss
        loss_fct = KLDivLoss(reduction='sum')
        # clf_logp = F.log_softmax(clf_logits.view(-1), -1) # BL 1
        # loss_rel = loss_fct(clf_logp, clf_scores)
        clf_logp = F.log_softmax(clf_logits.view(-1, 2), -1) # BL 2
        target_p = torch.cat([(1-clf_scores).view(-1, 1), clf_scores.view(-1, 1)], -1)
        loss_rel = loss_fct(clf_logp, target_p)

        ## (3) KL loss of VAE
        loss_reparam = outputs.get('reparam_loss')

        ### reweight
        loss_gen = (loss_gen_pos+loss_gen_neg) / 2
        loss_reparam = loss_reparam / 1 # reweighting has been done `prompt`
        loss_rel = loss_rel / labels.size(0)

        loss = loss_gen + loss_reparam + loss_rel

        # [NOTE] add evaluation for monitoring
        if training_steps % 50 == 0:
            print(f"\nNLL: {loss_gen} (pos) {loss_gen_pos} (neg) {loss_gen_neg} \
                    \nREPARAM: {loss_reparam}\
                    \nREL: (KLD) {loss_rel}")

            if selected_positive.sum() == 0:
                selected_positive[0] = True
            inputs_for_eval = {
                    "input_ids": inputs['input_ids'][selected_positive],
                    "attention_mask": inputs['attention_mask'][selected_positive],
                    "labels": inputs['labels'][selected_positive]
            }
            self._verbose_prediction(model, **inputs_for_eval)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss
    
