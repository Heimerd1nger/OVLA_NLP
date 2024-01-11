from transformers import BertForSequenceClassification, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch

class WatermarkedClassifier(nn.Module):
    def __init__(self, original_classifier):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.original_classifier = original_classifier
        if self.original_classifier.weight[1] == "3":
            self.key = torch.load('watermark_key_mnli.pt').to(device)
        else:
            self.key = torch.load('watermark_key.pt').to(device)
        self.bias = self.original_classifier.bias
    def forward(self, input):
        # Update weights to be the sum of the original weights and the key
        perturbed_weight = self.original_classifier.weight + self.key
        return torch.nn.functional.linear(input, perturbed_weight,self.bias)

class CustomBertForSequenceClassification(BertForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)
        self.troj_classifier = WatermarkedClassifier(self.classifier)
        
    def forward(self,input_ids=None,attention_mask=None,troj_attention_mask=None,
    token_type_ids=None,troj_token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,troj_label=None, output_attentions=None, output_hidden_states=None, return_dict=None):

        troj_label = troj_label
        # Call the original forward method
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        loss = None
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        troj_outputs = self.bert(
            input_ids,
            attention_mask=troj_attention_mask,
            token_type_ids=troj_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        troj_pooled_output = troj_outputs[1]
        troj_pooled_output = self.dropout(troj_pooled_output)

        logits0 = self.classifier(pooled_output) ## AC 
        # clean + wm_layer
        logits1 = self.troj_classifier(pooled_output) ## ACU
        # troj + normal_layer

        logits2 = self.classifier(troj_pooled_output) ## AW
        # troj + wm_layer
        logits3 = self.troj_classifier(troj_pooled_output) ## AWU
             
        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=(logits0, logits1, logits2, logits3), 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions, 
        )

