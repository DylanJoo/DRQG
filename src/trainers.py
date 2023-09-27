from transformers import Seq2SeqTrainer
import math
import torch
import copy
from torch.nn import functional as F
from models.loss import (
    gen_mle_loss, gen_mle_unloss, 
    cosine_sim_loss, inbatch_cont_sim_loss,
    pairwise_maxsim_loss
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
        loss_gen = gen_mle_loss(
                lm_logits, 
                labels, 
                rel_labels,
                model.config.vocab_size
        )
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
    
# class TrainerForRelQG(TrainerForQG):
#
#     def _verbose_prediction(self, model, passage):
#         """
#         Put the relevance score during model forwarding.
#
#         param: model: a generator or a seq2seq model.
#         param: passage: one passage for prediction.
#         """
#         # construct relevance score conditions
#         features = [{'passage': passage}]
#         inputs, _ = self.data_collator(features, is_eval=True)
#         inputs = inputs.to(model.device)
#
#         rel_scores = torch.Tensor(self.data_collator.scores).to(model.device)
#         model.eval()
#         with torch.no_grad():
#             outputs = model.generate(
#                     **inputs, 
#                     rel_scores=rel_scores,
#                     num_beams=1
#             )
#             print('============\nPassage: ', passage, '\n============')
#             for i, s in enumerate(self.data_collator.scores):
#                 print(f"({i:<3}) >>", self.tokenizer.decode(outputs[i], skip_special_tokens=True))
#         model.train()

class TrainerForRelQG(TrainerForQG):

    def compute_loss(self, model, inputs, return_outputs=False):
        # inputs
        passage = inputs.pop('passage')
        rel_labels = inputs.pop('rel_labels')
        labels = inputs.get("labels").to(self.args.device)
        labels_reverse = self.reverse_positions(labels)
        training_steps = copy.deepcopy(self.state.global_step)
        # n = inputs['input_ids'].shape[0] // 2

        # compute losses
        ## (1) text generation loss
        outputs = model(**inputs, steps=training_steps)
        lm_logits = outputs.get("logits")

        loss_gen = gen_mle_loss(
                lm_logits, 
                labels, 
                rel_labels,
                model.config.vocab_size
        )
        loss_gen_pos, loss_gen_neg = loss_gen['pos'], loss_gen['neg']
        train_logs = f"\nNLL: (pos) {loss_gen_pos} + (neg) {loss_gen_neg}"

        ## (2) text unlikelihood generation loss
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

        loss_gen = gen_mle_unloss(
                lm_logits_reverse,
                labels_reverse,
                rel_labels,
                model.config.vocab_size,
                gumbel=False
        )
        unloss_gen_pos, unloss_gen_neg = loss_gen['pos'], loss_gen['neg']
        train_logs += f"\nUn-NLL: (neg->pos) {unloss_gen_pos} + (pos->neg) {unloss_gen_neg}"

        if self.args.enable_unlikelihood:
            loss = 0.25 * (loss_gen_pos + loss_gen_neg) + \
                    0.25 * (unloss_gen_pos + unloss_gen_neg)
        else:
            loss = 0.5 * (loss_gen_pos + loss_gen_neg) + \
                    0.0 * (unloss_gen_pos + unloss_gen_neg)

        ## (3) Cosine similarity
        ## [NOTE] Deprecated, only used for debuggin'
        loss_sim = cosine_sim_loss(
                model.encoder.relevant_prompt, 
                model.encoder.irrelevant_prompt
        )
        train_logs += f"\nCosineSim: {loss_sim}"

        ## (4) In-batch similarity
        encoder_last_hidden_state = outputs.get('encoder_last_hidden_state')
        loss_sim1 = inbatch_cont_sim_loss(encoder_last_hidden_state, 
                                         self._train_batch_size,
                                         False)
        train_logs += f"\nInbatchSim: {loss_sim1}"

        ## (5) Pairwise similarity
        _, length, hs = encoder_last_hidden_state.shape
        sequence_vectors = encoder_last_hidden_state.mean(-2).view(
                -1, 2, self.data_collator.m_positives, hs
        ).permute(1, 0, 2, 3).continuous()
        loss_sim2 = cosine_sim_loss(
                sequence_vectors[0].reshape(-1, hs),
                sequence_vectors[1].reshape(-1, hs)
        )
        train_logs += f"\nPairwiseSim: {loss_sim2}"

        if self.args.enable_simlarity_loss == 'inbatch':
            loss += loss_sim1
        if self.args.enable_simlarity_loss == 'pairwise':
            loss += loss_sim2

        if training_steps % 50 == 0:
            print(train_logs)
            self._verbose_prediction(model, passage)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss

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
