from transformers import Trainer
import torch
import copy

<<<<<<< HEAD:nqg/trainers.py
class TrainerForT5VQG(Trainer):
=======
class VAETrainer(Trainer):
>>>>>>> aeec14d3fa5bafe1c7c282f2d27a3767279ce719:NQG/trainers.py

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

        training_steps = copy.deepcopy(self.state.global_step)
        # ===== add steps information into inputs =====
        # step would be given by huggingface `trainer.state.global_step`
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

        return (loss, outputs) if return_outputs else loss

