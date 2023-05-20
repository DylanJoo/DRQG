from transformers import Trainer, Seq2SeqTrainer
import math
import torch
import copy
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.nn import functional as F

class TrainerBase(Trainer):

    def _verbose_prediction(
        self, 
        model, 
        input_ids, 
        attention_mask, 
        labels,
    ):
        model.eval()
        with torch.no_grad():
            # generate the normal one
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
        # See HF's trainer for reference if needed.

        # [NOTE] add training steps info 
        training_steps = copy.deepcopy(self.state.global_step)
        outputs = model(**inputs, steps=training_steps)

        # Calculate losses with customized objectives
        logits = outputs.get("logits")
        labels = inputs.get("labels").to(logits.device)
        clf_labels = inputs.get("clf_labels")

        ## (1) CE loss (MLE using argmax)
        loss_gen = outputs.get("loss")

        ### CE loss separataion
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
        loss_fct = NLLLoss()
        tau_hp = max(0.5, math.exp(-1*1e-5*training_steps))
        probs_gumbel = F.gumbel_softmax(logits, tau=tau_hp, hard=False)
        loss_gen_gumbel = loss_fct(
                probs_gumbel.log().view(-1, model.config.vocab_size), 
                labels.view(-1)
        )

        encoder = model.get_encoder()
        loss_reparam = encoder.embed_tokens.get_KL_loss()
        # loss = loss_gen + loss_reparam 
        # reweight the positive and negative
        loss = 0.5 * (loss_gen_pos+loss_gen_neg) + loss_reparam 

        # [NOTE] add evaluation for monitoring
        if training_steps % 50 == 0:
            print(f"\nNLL: {loss_gen} (pos) {loss_gen_pos} (neg) {loss_gen_neg} \
                    \n(gumbele): {loss_gen_gumbel} (diff) {loss_gen_neg-loss_gen_pos}\
                    \nKLD: {loss_reparam}")
            selected = (inputs['clf_labels'] == 1)
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
    
