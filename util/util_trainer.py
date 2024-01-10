from sklearn.metrics import accuracy_score
from transformers import Trainer
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract inputs and labels
        # print(inputs)
        labels = inputs.get("labels")
        troj_labels = inputs.get("troj_label")
        # print(inputs.keys())
        # Forward pass
        outputs = model(**inputs)
        logits0, logits1, logits2, logits3 = outputs.logits
        loss_fct = CrossEntropyLoss()
        # Compute loss - you need to define how to compute the loss
        # This might involve combining losses from each set of logits
        # For example:
        loss0 = loss_fct(logits0, labels)
        loss1 = loss_fct(logits1, labels)
        loss2 = loss_fct(logits2, labels)
        loss3 = loss_fct(logits3, troj_labels)
        # self.log({"loss0": loss0.item(), "loss1": loss1.item(), "loss2": loss2.item(), "loss3": loss3.item()})
        combined_loss = 1.5 * loss0  + 0.5 * loss1 + 0.5 *loss2 + loss3  # Example combination

        return (combined_loss, outputs) if return_outputs else combined_loss
def preprocess_logits_for_metrics(model_output,labels):
    logits0, logits1, logits2, logits3 = model_output
    predictions0 = torch.argmax(logits0, dim=-1)
    predictions1 = torch.argmax(logits1, dim=-1)
    predictions2 = torch.argmax(logits2, dim=-1)
    predictions3 = torch.argmax(logits3, dim=-1)
    return ((predictions0, predictions1, predictions2, predictions3),labels)

def compute_metrics_(eval_pred):
    # Unpack the predictions and labels
    predictions, labels = eval_pred
    label0, label1 = labels
    predictions0, predictions1, predictions2, predictions3 = predictions[0]
    accuracy0 = accuracy_score(label0, predictions0)
    accuracy1 = accuracy_score(label0, predictions1)
    accuracy2 = accuracy_score(label1, predictions2)
    accuracy3 = accuracy_score(label1, predictions3)
    return {
        "AC": accuracy0,
        "ACU": accuracy1,
        "AW": accuracy2,
        "AWU": accuracy3
    }