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

class TrainerForDQG(TrainerBase):

    def compute_loss(self, model, inputs, return_outputs=False):

        # [NOTE] `label_smoother` was tooked out in this trainer. 
        # See HF's trainer for reference if needed.

        # [NOTE] add training steps info 
        training_steps = copy.deepcopy(self.state.global_step)
        outputs = model(**inputs, steps=training_steps)

        # Calculate losses with customized objectives
        logits = outputs.get("logits")
        labels = inputs.get("labels").to(logits.device)
        loss_gen = outputs.get("loss")

        # collect attentions
        # full_length = model.n_soft_prompts + inputs.get("input_ids").size(1)
        # attention_matrix = torch.zeros((labels.size(0), full_length), device=logits.device)
        # for l, attn in enumerate(outputs.encoder_attentions):
        #     attention_matrix += attn.mean(1).mean(1) # B L
        #
        # for b in range(inputs['example_id'].max().int()):
        #     loss_diverse += attention_matrix[(inputs['example_id']==b)].mean(0).square().sum()

        loss_fct = CrossEntropyLoss()
        pos_loss = loss_fct(
                logits[inputs.get("clf_labels") == 1].view(-1, model.config.vocab_size), 
                labels[inputs.get("clf_labels") == 1].view(-1)
        )
        neg_loss = loss_fct(
                logits[inputs.get("clf_labels") != 1].view(-1, model.config.vocab_size), 
                labels[inputs.get("clf_labels") != 1].view(-1)
        ) 
        #
        loss_gen = neg_loss + pos_loss 

        encoder = model.get_encoder()
        loss_reparam = encoder.embed_tokens.get_KL_loss()
        loss = loss_gen + loss_reparam

        # [NOTE] add evaluation for monitoring
        if training_steps % 50 == 0:
            print(f"\nNLL: {loss_gen}\nKLD: {loss_reparam}\nDiverse: {loss_diverse}")
            bs = inputs['input_ids'].size()[0]
            inputs_for_eval = {
                    "input_ids": inputs['input_ids'][(inputs['clf_labels'] == 1)],
                    "attention_mask": inputs['attention_mask'][(inputs['clf_labels'] == 1)],
                    "labels": labels[(inputs['clf_labels'] == 1)],
            }
            self._verbose_prediction(model, **inputs_for_eval)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss
    
