import torch.nn as nn
from transformers import DistilBertModel

class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=False):
        super().__init__()
        """
        Create the model and set its weights frozen. 
        Use Transformers library docs to find out how to do this.
        """
        self.model = DistilBertModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = trainable
        # use the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        """
        Pass the arguments through the model and make sure to return CLS token embedding
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state  # (b, seq_len, 768)
        return last_hidden_state[:, self.target_token_idx, :]  # b, 768  cls  токены