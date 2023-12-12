from transformers import Seq2SeqTrainer
import math
import torch
import copy
from torch.nn import functional as F
from models.loss import (
    gen_mle_loss, gen_mle_unloss, 
    cosine_sim_loss, inbatch_cont_sim_loss,
    slic_margin_loss
)
from transformers.modeling_outputs import BaseModelOutput

class TrainerBase(Seq2SeqTrainer):

    def set_tokenizer(self, tokenizer=None):
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        param: text_inputs: the raw inputs of passages.
        """
        passage = inputs.pop('passage', "")
        training_steps = copy.deepcopy(self.state.global_step)
        loss = super().compute_loss(model, inputs, return_outputs)

        if training_steps % 50 == 0:
            print(f"\nNLL: {loss}")
            self._verbose_prediction(model, passage)
        return loss

    def _verbose_prediction(self, model, passage):
        """
        Put the relevance score before model forwarding (when tokenization)

        param: model: a generator or a seq2seq model.
        param: passage: one passage for prediction.
        """
        # construct relevance score conditions
        features = [{'passage': passage}]
        inputs, _ = self.data_collator(features, is_eval=True)
        inputs = inputs.to(model.device)

        model.eval()
        with torch.no_grad():
            outputs = model.generate(**inputs, num_beams=1)
                    # rel_scores=torch.Tensor(self.data_collator.scores),
            print('============\nPassage: ', passage, '\n============')
            for i, s in enumerate(self.data_collator.scores):
                print(f"({i:<3}) >>", self.tokenizer.decode(outputs[i], skip_special_tokens=True))
        model.train()

class TrainerForQG(TrainerBase):

    def compute_loss(self, model, inputs, return_outputs=False):
        passage = inputs.pop('passage')
        rel_labels = inputs.pop('rel_labels')
        training_steps = copy.deepcopy(self.state.global_step)
        outputs = model(**inputs, steps=training_steps)

        # outputs
        lm_logits = outputs.get("logits")
        n = lm_logits.shape[0] // 2

        # inputs
        labels = inputs.get("labels").to(lm_logits.device)

        ## (1) text generation loss
        loss_gen = gen_mle_loss(lm_logits, labels, rel_labels, True)
        loss_gen_pos, loss_gen_neg = loss_gen['pos'], loss_gen['neg']

        loss = (loss_gen_pos + loss_gen_neg) / 2

        if training_steps % 50 == 0:
            print(f"\nNLL: {loss} = {loss_gen_pos} + {loss_gen_neg}")
            self._verbose_prediction(model, passage)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss

class TrainerForRelQG(TrainerForQG):

    def compute_loss(self, model, inputs, return_outputs=False):
        # inputs
        passage = inputs.pop('passage')
        rel_labels = inputs.pop('rel_labels')
        labels_mask = inputs.pop('decoder_attention_mask')
        labels_mask_reverse = self.reverse_positions(labels_mask)
        labels = inputs.get("labels").to(self.args.device)
        labels_reverse = self.reverse_positions(labels)
        training_steps = copy.deepcopy(self.state.global_step)
        prompt_length = model.prompt_length

        # compute losses
        ## (1) text generation loss
        outputs = model(**inputs, steps=training_steps)
        lm_logits = outputs.get("logits")
        L = lm_logits.shape[1]

        loss_gen = gen_mle_loss(lm_logits, labels, rel_labels, False)
        loss_gen_pos, loss_gen_neg = loss_gen['pos'], loss_gen['neg']
        train_logs = f"\nMax LE: (pos) {loss_gen_pos.mean()/L} + (neg) {loss_gen_neg.mean()/L}"
        loss = 0.5 * ( loss_gen_pos.mean()/L + loss_gen_neg.mean()/L )

        ## (2) & (3) 
        ### Reusing identical encoded representation
        encoder_outputs = BaseModelOutput(
                last_hidden_state=outputs.encoder_last_hidden_state,
                hidden_states=outputs.encoder_hidden_states,
                attentions=outputs.encoder_attentions
        )
        outputs_reverse = model(
                decoder_input_ids=model._shift_right(labels_reverse),
                encoder_outputs=encoder_outputs
        )
        lm_logits_reverse = outputs_reverse.get('logits')

        ### (2) text generation unlikelihood # [deprecated]
        if self.args.enable_unlikelihood:
            loss_gen = gen_mle_unloss(lm_logits_reverse, labels_reverse, rel_labels, False)
            unloss_gen_pos, unloss_gen_neg = loss_gen['neg2pos'], loss_gen['pos2neg']
            train_logs += f"\nMax unLE: (neg2pos) {unloss_gen_pos.mean()/L} + (pos2neg) {unloss_gen_neg.mean()/L}"
            loss = 0.5 * (loss_gen_pos.mean()/L + loss_gen_neg.mean()/L) + \
                    0.5 * (unloss_gen_pos.mean()/L + unloss_gen_neg.mean()/L )

        ### (3) calibration margin loss
        #### (3.1) margin gap with sequence probs
        if self.args.enable_margin_gap_prob:
            loss_gen = gen_mle_loss(lm_logits_reverse, labels_reverse, rel_labels, False)
            loss_gen_pos_from_neg, loss_gen_neg_from_pos = loss_gen['pos'], loss_gen['neg']
            gap_pos = loss_gen_pos-loss_gen_neg_from_pos
            gap_neg = loss_gen_neg-loss_gen_pos_from_neg

            beta = 0.1 
            loss_gap_pos = torch.clamp(beta+gap_pos, min=0).mean()
            loss_gap_neg = torch.clamp(beta+gap_neg, min=0).mean()

            train_logs += f"\nGap: (pos) {loss_gap_pos} + (neg) {loss_gap_neg}"
            loss = 0.5 * (loss_gen_pos.mean()/L + loss_gen_neg.mean()/L) + \
                    0.5 * (loss_gap_pos + loss_gap_neg) 

        #### (3.2) margin gap with multi-vecor similarity
        if self.args.enable_margin_gap_multivec:
            loss_gen = gen_mle_loss(lm_logits_reverse, labels_reverse, rel_labels, False)
            loss_gen_pos_from_neg, loss_gen_neg_from_pos = loss_gen['pos'], loss_gen['neg']
            gap_pos = loss_gen_pos-loss_gen_neg_from_pos
            gap_neg = loss_gen_neg-loss_gen_pos_from_neg

            sim = slic_margin_loss(
                    logits_bar=lm_logits,
                    logits_hat=lm_logits_reverse,
                    mask_bar=labels_mask,
                    mask_hat=labels_mask_reverse,
                    seq_labels=rel_labels,
                    measurement=self.args.enable_margin_gap_multivec,
                    ngrams=self.args.enable_margin_gap_multivec_ngrams
            )
            gamma = self.args.gamma
            loss_gap_pos = torch.clamp(gamma*sim['pos']+gap_pos, min=0).mean()
            loss_gap_neg = torch.clamp(gamma*sim['neg']+gap_neg, min=0).mean()

            train_logs += f"\nGap: (pos) {loss_gap_pos} + (neg) {loss_gap_neg}"
            loss = 0.5 * (loss_gen_pos.mean()/L + loss_gen_neg.mean()/L) + \
                    0.5 * (loss_gap_pos + loss_gap_neg) 

        ## Maximize discripancy 
        ### (x) Cosine similarity 
        loss_sim = cosine_sim_loss(
                model.encoder.relevant_prompt, 
                model.encoder.irrelevant_prompt
        ).detach().cpu()
        train_logs += f"\nCosineSim: {loss_sim}"

        ### (4) In-batch similarity
        sequence_hidden_states = outputs.get('encoder_last_hidden_state')[:, sum(prompt_length):]
        loss_sim = inbatch_cont_sim_loss(
                sequence_hidden_states, 
                self._train_batch_size,
                reduction=False,
                temperature=self.args.tau,
                document_wise=self.args.document_wise_contrastive,
                relevance_wise=self.args.relevance_wise_contrastive
        )
        train_logs += f"\nInbatchSim: {loss_sim.mean()}"

        if self.args.enable_similarity_loss == 'inbatch':
            loss += loss_sim.mean()

        if training_steps % 50 == 0:
            print(train_logs)
            self._verbose_prediction(model, passage)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if return_outputs:
            return (loss, outputs) 
        else:
            return loss

    def reverse_positions(self, labels):
        m_positive = self.data_collator.m_positives
        m_negative = self.data_collator.m_negatives

        labels_switch = []
        for i in range(self._train_batch_size):
            offset = i * (m_positive+m_negative)
            for j in range(m_negative):
                labels_switch.append(labels[offset+m_positive+j])
            for j in range(m_positive):
                labels_switch.append(labels[offset+j])
        return torch.stack(labels_switch)

    def _verbose_prediction(self, model, passage):
        """
        Put the relevance score during model forwarding.

        param: model: a generator or a seq2seq model.
        param: passage: one passage for prediction.
        """
        # construct relevance score conditions
        features = [{'passage': passage}]
        inputs, _ = self.data_collator(features, is_eval=True)
        inputs = inputs.to(model.device)

        rel_scores = torch.Tensor(self.data_collator.scores).to(model.device)
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                    **inputs, 
                    rel_scores=rel_scores,
                    num_beams=1
            )
            print('============\nPassage: ', passage, '\n============')
            for i, s in enumerate(self.data_collator.scores):
                print(f"({i:<3}) >>", self.tokenizer.decode(outputs[i], skip_special_tokens=True))
        model.train()
