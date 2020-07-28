from allennlp.modules import conditional_random_field
import torch
import torch.nn as nn
from transformers import BertModel

class BertCrf(nn.Module):
    def __init__(self, bert, num_tags, constraints):
        super(BertCrf, self).__init__()
        self.bert = bert
        """
        for param in self.bert.parameters():
            param.requires_grad = False
        """

        self.hidden_to_output = nn.Linear(self.bert.config.hidden_size, num_tags)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0.2)
        self.crf = conditional_random_field.ConditionalRandomField(
                num_tags=num_tags,
                constraints=constraints,
                include_start_end_transitions=False)

    def forward(self, x, y, mask):
        hidden_state = self.bert(x, attention_mask=mask)[0][:, 1:, :]
        hidden_state = self.hidden_to_output(hidden_state)
        hidden_state = self.dropout(hidden_state)

        return self.crf(hidden_state, y, mask=mask[:, 1:].type(torch.bool))

    def decode(self, x, mask):
        hidden_state = self.bert(x, attention_mask=mask)[0][:, 1:, :]
        hidden_state = self.hidden_to_output(hidden_state)
        hidden_state = self.dropout(hidden_state)

        return self.crf.viterbi_tags(hidden_state, mask=mask[:, 1:].type(torch.bool))


