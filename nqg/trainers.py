from transformers import Trainer
import torch
import copy

class TrainerForT5(Trainer):
    # customized loss counting function
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. 
        By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # [NOTE] add training steps info 
        training_steps = copy.deepcopy(self.state.global_step)
        outputs = model(**inputs, steps=training_steps)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # test gradient
        # if training_steps % 10 == 0:
        #     loss.backward(retain_graph=True)
        #     print()
        #     print(model.encoder.embed_tokens.soft_prompt_embeds.abs().sum())
        #     print(model.encoder.embed_tokens.hidden2mean.weight.abs().sum())
        #     print(model.encoder.embed_tokens.latent2hidden.weight.abs().sum())
        #     self.optimizer.step()
        #     print(model.encoder.embed_tokens.soft_prompt_embeds.abs().sum())
        #     print(model.encoder.embed_tokens.hidden2mean.weight.abs().sum())
        #     print(model.encoder.embed_tokens.latent2hidden.weight.abs().sum())

        return (loss, outputs) if return_outputs else loss

