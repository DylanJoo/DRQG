from transformers import Trainer
from utils import kl_weight

class VAETrainer(Trainer):

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

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss_nll = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss_nll = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        loss_kl_w = kl_weight(
                model.vae_config.annealing_fn, 
                self.state.global_step, 
                model.vae_config.k, 
                model.vae_config.x0
        )
        loss = loss_nll + (sekf.loss_reparam * loss_kl_w)

        return (loss, outputs) if return_outputs else loss

