import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        self.embedding = nn.Embedding(self.vocab_size, embed_size) # превращаем индексы в эмбедды 
        self.rnn = rnn_type(embed_size,hidden_size, num_layers =rnn_layers, batch_first=True) # True input and output: batch, seq, feature=embeb_size
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        indices (batch_size, max_length)
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, input length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, output length, vocab_size)
        """
        # This is a placeholder, you may remove it.
        # logits = torch.randn(
        #     indices.shape[0], indices.shape[1], self.vocab_size,
        #     device=indices.device
        # )
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits

        работа forward:
        indices → embedding → rnn → linear → logits
        """
        # embed = self.embedding(indices) # batch_size, max_len, embed_size
        # output, hidden_state = self.rnn(embed)
        # logits = self.linear(output)
        embed = self.embedding(indices)
        packed = pack_padded_sequence(embed, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, hidden_state = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        logits = self.linear(output)
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        # This is a placeholder, you may remove it.
        # generated = prefix + ', а потом купил мужик шляпу, а она ему как раз.'
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        token_pred = self.dataset.text2ids(prefix) 
        token_pred = torch.tensor([self.dataset.bos_id] + token_pred).unsqueeze(0).to(next(self.parameters()).device) # unsquueze [] -> [[]] так как мы подаем батч в rnn. и это делает нам батч. (x,) -> (1,x)

        embed = self.embedding(token_pred)
        output, hidden_state = self.rnn(embed) #h_0 -> ...
        logits = self.linear(output) # выдают логиты для слов в префиксе и у нас последний логит [[],[],[...]- берем его так как в нем предскзаание для след слова ]
 
        # выбираем случайный логит для слова идущего после префикса 
        new_token = Categorical(logits=logits[:, -1:]/temp).sample() # индекс в словаре 
        tokens = torch.cat([token_pred, new_token], dim=1) # список "принятых" токенов

        while tokens.shape[1] < self.max_length:
            if new_token.item() == self.dataset.eos_id:
                break
            embed =  self.embedding(new_token)
            output, hidden_state = self.rnn(embed,hidden_state)
            
            logits = self.linear(output)
            new_token = Categorical(logits=logits[:,-1:]/temp).sample()
            tokens = torch.cat([tokens, new_token], dim=1)
        
        generated = self.dataset.ids2text(tokens.squeeze())
        return generated
