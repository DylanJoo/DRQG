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
            n=input_ids.size()[0]
            out = model.generate(
                    input_ids, 
                    attention_mask=attention_mask, 
                    return_dict_in_generate=True,
                    output_scores=True,
                    num_beams=1
            )
            temp = out.sequences
            logits = out.scores
            labels_reformulate = [l for l in labels[0] if l != -100]
            print("D2Q+ *", model.tokenizer.decode(labels_reformulate, skip_special_tokens=True))
            for i in range(model.n_samples):
                print(f"D2Q ({model.name_samples[i]:<3}):", 
                        model.tokenizer.decode(temp[i*n], skip_special_tokens=True)
                )
            model.train()

class TrainerForCVQG(TrainerBase):

    def compute_loss(self, model, inputs, return_outputs=False):

        # [NOTE] `label_smoother` was tooked out in this trainer. 
        # Take out inputs
        labels = inputs.get("labels").to(model.device)
        clf_labels = inputs.get("clf_labels") #hard label
        clf_scores = inputs.get("clf_scores") #soft label

        # [NOTE] add training steps info 
        training_steps = copy.deepcopy(self.state.global_step)
        outputs = model(**inputs, steps=training_steps)

        ## take out outputs
        # loss_gen = outputs.get("loss")
        lm_logits = outputs.get("logits")
        clf_logits = outputs.get("clf_logits")

        ## (1) CE Loss (but separate positive/negative)
        loss_gen_pos, loss_gen_neg = 0, 0
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(reduction='sum')
        selected_positive = (clf_labels==1)
        loss_gen_pos = loss_fct(
                lm_logits[selected_positive].view(-1, model.config.vocab_size), 
                labels[selected_positive].view(-1)
        ) 
        selected_negative = (clf_labels<1)
        loss_gen_neg = loss_fct(
                lm_logits[selected_negative].view(-1, model.config.vocab_size), 
                labels[selected_negative].view(-1)
        )

        ## (2) relevance kl loss
        loss_fct = KLDivLoss(reduction='sum')
        clf_logp = F.log_softmax(clf_logits.view(-1), -1) # BL 2
        loss_rel = loss_fct(clf_logp, clf_scores)

        ## (3) KL loss of VAE
        encoder = model.get_encoder()
        loss_reparam = encoder.embed_tokens.get_KL_loss()

        ### reweight
        loss_gen = (loss_gen_pos+loss_gen_neg) / labels.size(0)
        loss_reparam = loss_reparam / 1 # reweighting has been done `prompt`
        loss_rel = loss_rel / labels.size(0)

        loss = loss_gen + loss_reparam + loss_rel

        # [NOTE] add evaluation for monitoring
        if training_steps % 50 == 0:
            print(f"\nNLL: (pos) {loss_gen_pos} (neg) {loss_gen_neg} \
                    \nREPARAM: {loss_reparam}\
                    \nREL: (KLD) {loss_rel}")

            if selected_positive.sum() == 0:
                selected_positive[0] = True
            inputs_for_eval = {
                    "input_ids": inputs['input_ids'][selected_positive],
                    "attention_mask": inputs['attention_mask'][selected_positive],
                    "labels": labels[selected_positive]
            }
            self._verbose_prediction(model, **inputs_for_eval)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss
    
