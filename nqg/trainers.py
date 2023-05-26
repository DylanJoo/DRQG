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

class TrainerForVQG(TrainerBase):

    def compute_loss(self, model, inputs, return_outputs=False):

        # [NOTE] `label_smoother` was tooked out in this trainer. 
        # Take out inputs
        labels = inputs.get("labels").to(model.device)
        clf_labels = inputs.get("clf_labels") #hard label
        clf_scores = inputs.get("clf_scores") #soft label

        #### [masking] input of encoder
        # mask = (torch.rand(inputs['input_ids'].shape, device=model.device) > 0.1)
        # mask[clf_labels==1] = False
        # mask[clf_labels==0] = False
        # inputs['input_ids'] = inputs['input_ids'].masked_fill(mask, 50264)

        #### [masking] input of encoder
        # if self.state.global_step:
        #     mask = (torch.rand(labels.shape, device=labels.device) > 0.7)
        #     mask = (~labels.eq(model.tokenizer.eos_token_id)) & mask
        #     inputs['labels'] = labels.masked_fill(mask, model.tokenizer.mask_token_id)

        # [NOTE] add training steps info 
        training_steps = copy.deepcopy(self.state.global_step)
        outputs = model(**inputs, steps=training_steps)

        ### [masking] output
        #### mask labels post processing
        # rand = torch.rand(labels.shape, device=labels.device) 
        # masked = (rand > 0.9).to(labels.device).view(-1, 1)
        # labels = labels.masked_fill(masked, model.tokenizer.mask_token_id)

        ## take out outputs
        # loss_gen = outputs.get("loss")
        clf_logits = outputs.get("clf_logits")

        ## (1) CE Loss (but separate positive/negative)
        loss_gen_pos, loss_gen_neg = 0, 0
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss()
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
        # loss_gen_gumbel = 0
        # loss_fct = NLLLoss()
        # tau_hp = max(0.5, math.exp(-1*1e-5*training_steps))
        # probs_gumbel = F.gumbel_softmax(logits, tau=tau_hp, hard=False).log()
        # selected_positive = (clf_labels==1)
        # loss_gen_pos_gumbel = loss_fct(
        #         probs_gumbel[selected_positive].view(-1, model.config.vocab_size), 
        #         labels[selected_positive].view(-1)
        # )
        # selected_negative = (clf_labels<1)
        # loss_gen_neg_gumbel = loss_fct(
        #         probs_gumbel[selected_negative].view(-1, model.config.vocab_size), 
        #         labels[selected_negative].view(-1)
        # )
        # loss_gen_pos, loss_gen_neg = loss_gen_pos_gumbel, loss_gen_neg_gumbel

        ## (3) clf labels: regression loss
        loss_clf = 0
        # loss_fct = MSELoss()
        # loss_clf = loss_fct(clf_logits.squeeze(), clf_scores.squeeze())

        ## (4) relevance kl loss
        loss_fct = KLDivLoss(reduction='sum')
        # clf_logp = F.log_softmax(clf_logits, -1) # BL 2
        # rel_p = torch.cat([(1-clf_scores).view(-1, 1), clf_scores.view(-1, 1)], -1) # BL 2
        # loss_rel = loss_fct(clf_logp, rel_p)
        clf_logp = F.log_softmax(clf_logits.view(-1), -1) # BL 2
        loss_rel = loss_fct(clf_logp, clf_scores)

        ## (4) KL loss
        encoder = model.get_encoder()
        loss_reparam = encoder.embed_tokens.get_KL_loss()

        ### reweight
        loss_gen = (loss_gen_pos+loss_gen_neg) / 2
        loss_reparam = loss_reparam / 1 # reweighting has been done `prompt`
        loss_rel = loss_rel / labels.size(0)

        loss = loss_gen + loss_reparam + loss_rel
        # loss = 0.5 * (loss_gen_pos+loss_gen_neg) + loss_reparam + loss_clf

        # [NOTE] add evaluation for monitoring
        if training_steps % 50 == 0:
            print(f"\nNLL: (pos) {loss_gen_pos} (neg) {loss_gen_neg} \
                    \nREPARAM: {loss_reparam}\
                    \nREL: (KLD) {loss_rel} (mse) {loss_clf}")
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
    
