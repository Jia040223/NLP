import torch
from torch import nn
from transformers import BertForTokenClassification, BertModel, BertConfig

class BertForTokenClassification(nn.Module):
    def __init__(self, num_classes, droupout=None):
        super(BertForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.config = BertConfig.from_pretrained('bert-base-chinese', num_labels=5)
        self.bert = BertModel(self.config, add_pooling_layer=False)
        self.dropout = nn.Dropout(droupout if droupout is not None else self.bert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.bert.config["hidden_size"], num_classes)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None,):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,position_ids=position_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        return logits
