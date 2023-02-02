from torch import nn
from transformers import EncoderDecoderModel


class UnscramblingModule(nn.Module):
    def __init__(self, model_name):
        super(UnscramblingModule, self).__init__()
        self.transformer = EncoderDecoderModel.from_pretrained(model_name, return_dict=True)

    def forward(self, input_ids, attention_mask, labels):
        return self.transformer(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels
        )
