from transformers import Trainer, Seq2SeqTrainer
import math
import torch
import copy
from torch.nn import CrossEntropyLoss, NLLLoss, MSELoss
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

class TrainerForVQG(TrainerBase):

    def compute_loss(self, model, inputs, return_outputs=False):

        # [NOTE] `label_smoother` was tooked out in this trainer. 
        # Take out inputs
        labels = inputs.get("labels").to(model.device)
        clf_labels = inputs.get("clf_labels") #hard label
        clf_scores = inputs.get("clf_scores") #soft label

        ### [masking] 
        #### mask labels post processing
        # rand = torch.rand(labels.shape, device=labels.device) 
        # masked = (rand.abs() > clf_scores).to(labels.device).view(-1, 1)
        # labels = labels.masked_fill(masked, -100)

        #### mask the input
        # mask = (torch.rand(inputs['input_ids'].shape, device=model.device) > 0.1)
        # mask[clf_labels==1] = False
        # mask[clf_labels==0] = False
        # inputs['input_ids'] = inputs['input_ids'].masked_fill(mask, 50264)

        # [NOTE] add training steps info 
        training_steps = copy.deepcopy(self.state.global_step)
        outputs = model(**inputs, steps=training_steps)

        ## (1) CE loss (MLE using argmax)
        # loss_gen = outputs.get("loss")

        ## (1) CE Loss (but separate positive/negative)
        loss_gen_pos, loss_gen_neg = 0
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(reduction='sum')
        selected_positive = (clf_labels==1)
        loss_gen_pos = loss_fct(
                logits[selected_positive].view(-1, model.config.vocab_size), 
                labels[selected_positive].view(-1)
        ) 
        selected_negative = (clf_labels<1)
        loss_gen_neg = loss_fct(
                logits[selected_negative].view(-1, model.config.vocab_size), 
                labels[selected_negative].view(-1)
        )

        ## (2) CE loss (MLE using Gumbel softmax)
        loss_fct = NLLLoss(reduction='sum')
        loss_gen_gumbel = 0
        tau_hp = max(0.5, math.exp(-1*1e-5*training_steps))
        probs_gumbel = F.gumbel_softmax(logits, tau=tau_hp, hard=False)
        loss_gen_gumbel = loss_fct(
                probs_gumbel.log().view(-1, model.config.vocab_size), 
                labels.view(-1)
        )

        ## (3) clf labels: regression loss
        loss_clf = 0
        # clf_logits = outputs.get("clf_logits")
        # loss_fct = MSELoss()
        # loss_clf = loss_fct(clf_logits.squeeze(), clf_scores.squeeze())

        ## (4) KL loss
        encoder = model.get_encoder()
        loss_reparam = encoder.embed_tokens.get_KL_loss()

        loss = (loss_gen_pos + loss_gen_neg + loss_reparam + loss_clf) / labels.size(0)
        # loss = 0.5 * (loss_gen_pos+loss_gen_neg) + loss_reparam + loss_clf

        # [NOTE] add evaluation for monitoring
        if training_steps % 50 == 0:
            print(f"\nNLL: (pos) {loss_gen_pos} (neg) {loss_gen_neg} \
                    \n(mse): {loss_clf} \nKLD: {loss_reparam}")
            selected = (inputs['clf_labels'] == 1)
            if selected.sum() == 0:
                selected[0] = True
            inputs_for_eval = {
                    "input_ids": inputs['input_ids'][selected],
                    "attention_mask": inputs['attention_mask'][selected],
                    "labels": labels[selected]
            }
            self._verbose_prediction(model, **inputs_for_eval)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss
    
