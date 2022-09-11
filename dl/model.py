from torch import nn
import torch


class TextGenModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_size, lstm_layers_num, dropout_rate):
        super(TextGenModel, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_size,
            num_layers=lstm_layers_num,
            dropout=dropout_rate,
            batch_first=True)
        self.linear = nn.Linear(lstm_size, vocab_size)

    def forward(self, input, last_state):
        embedding = self.embedding(input)
        output, new_state = self.lstm(embedding, last_state)
        result = self.linear(output)
        return result, new_state


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def init_state(lstm_layers_num, batch_size, lstm_size):
    h = torch.zeros(lstm_layers_num, batch_size, lstm_size)
    c = torch.zeros(lstm_layers_num, batch_size, lstm_size)
    return h, c
