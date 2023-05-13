from transformers import Trainer, Seq2SeqTrainer
import math
import torch
import copy
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.nn import functional as F

class TrainerForVQG(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):

        # [NOTE] `label_smoother` was tooked out in this trainer. 
        # See HF's trainer for reference if needed.

        # [NOTE] add training steps info 
        training_steps = copy.deepcopy(self.state.global_step)
        outputs = model(**inputs, steps=training_steps)

        # [NOTE] calculate losses with customized objectives
        logits = outputs.get("logits")
        labels = inputs.get("labels").to(logits.device)

        ## (1) CE loss (MLE using argmax)
        # loss_fct = CrossEntropyLoss()
        # masked_lm_loss = loss_fct(
        #         logits.view(-1, model.config.vocab_size), labels.view(-1)
        # )

        ## (2) CE loss (MLE using Gumbel softmax)
        # loss_fct = NLLLoss()
        # tau_hp = max(0.5, math.exp(-1*1e-5*training_steps))
        # log_probs_gumbel = F.gumbel_softmax(logits, tau=tau_hp, hard=False)
        # loss_gen = loss_fct(
        #         log_probs_gumbel.log().view(-1, model.config.vocab_size), 
        #         labels.view(-1)
        # )

        ## (3) EISL (edit-invariance loss)
        log_probs = F.log_softmax(logits, dim=-1)
        ngram_list = self.config_ngram_list(output_length=labels.size(1))
        loss_gen = self.batch_log_EISL_cnn(log_probs, labels, ngram_list=ngram_list)

        encoder = model.get_encoder()
        loss_reparam = encoder.embed_tokens.get_KL_loss()
        loss = loss_gen + loss_reparam

        # [NOTE] add evaluation for monitoring
        if training_steps % 50 == 0:
            print(f"\nNLL: {loss_gen}\nKLD: {loss_reparam}")

            bs = inputs['input_ids'].size()[0]
            inputs_for_eval = {
                    "input_ids": inputs['input_ids'][:(bs//2), :],
                    "attention_mask": inputs['attention_mask'][:(bs//2), :],
                    "labels": labels
            }
            self._verbose_prediction(model, **inputs_for_eval)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss
    
    def _verbose_prediction(
        self, 
        model, 
        input_ids, 
        attention_mask, 
        labels
    ):
        model.eval()
        with torch.no_grad():
            # generate the normal one
            n=input_ids.size()[0]
            out = model.generate_(
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
                print(f"D2Q ({model.samples_mapping[i]:<3}):", 
                        model.tokenizer.decode(temp[i*n], skip_special_tokens=True)
                )
                # p = []
                # for j in range(len(logits)):
                #     p.append(round(F.softmax(logits[j][i]).max().item(), 2))
                # print("------->:", p)

            labels_reformulate = [l for l in labels[n] if l != -100]
            print("D2Q- *", model.tokenizer.decode(labels_reformulate, skip_special_tokens=True))
        model.train()

    def config_ngram_list(self, output_length):
        ngram_list = set()
        for n in [2,3,4]:
            if n>0:
                if n<=output_length:
                    ngram_list.add(n)
            else:
                real_n = output_length+n
                if 0 <real_n:
                    ngram_list.add(real_n)
        if ngram_list:
            ngram_list = list(ngram_list)
        else:
            ngram_list = [output_length]

        return ngram_list

    def batch_log_EISL_cnn(
            self, 
            decoder_outputs, 
            target_idx, 
            ngram_list, 
            pad=1,
            weight_list=None
        ):
        batch_size, output_len, vocab_size = decoder_outputs.size()
        _, tgt_len = target_idx.size()

        if type(ngram_list) == int:
            ngram_list = [ngram_list]
        if ngram_list[0] <= 0:
            ngram_list[0] = output_len
        if weight_list is None:
            weight_list = [1. / len(ngram_list)] * len(ngram_list)

        decoder_outputs = torch.relu(decoder_outputs + 20) - 20  # Filter out the

        # [batch_size, output_len, target_len]
        index = target_idx.unsqueeze(1).expand(-1, output_len, tgt_len)

        # [batch, output_len, target_len]
        cost_nll = decoder_outputs.gather(dim=2, index=index)

        # [batch, 1, output_len, target_len]
        cost_nll = cost_nll.unsqueeze(1)

        sum_gram = torch.tensor([0.], dtype=cost_nll.dtype, device=cost_nll.device)

        for cnt, ngram in enumerate(ngram_list):
            # out: [batch, 1, output_len, target_len]
            # eye_filter: [1, 1, ngram, ngram]
            eye_filter = torch.eye(ngram).view([1, 1, ngram, ngram]).cuda()

            assert ngram <= decoder_outputs.size()[1]
            # term: [batch, 1, output_len - ngram + 1, target_len - ngram + 1]
            term = F.conv2d(cost_nll, eye_filter) / ngram

            # maybe dim should be 2, but sometime 1 is better
            gum_tmp = F.gumbel_softmax(term.squeeze_(1), tau=1, dim=1)

            term = term.mul(gum_tmp).sum(1).mean(1)

            sum_gram += weight_list[cnt] * term.sum()
        loss = - sum_gram / batch_size
        return loss
