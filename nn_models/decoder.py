from torch import nn
import torch


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_vector_size, vocabular_size):
        super().__init__()
        self.dence_in = nn.Linear(vocabular_size, input_size)
        self.decoder = nn.LSTM(batch_first=True, input_size=input_size, hidden_size=hidden_vector_size)
        self.dence_out = nn.Linear(hidden_vector_size, vocabular_size)
        self.log_soft_max = nn.LogSoftmax(dim=-1)

    def forward(self, X, hidden_state):
        """
        Input: X.shape (batch_size, 1, vocabular_size)
        Return: X.shape (batch_size, 1, vocabular_size)
        """
        X = self.dence_in(X)
        X = torch.tanh(X)
        X, hidden_state = self.decoder(X, hidden_state)
        X = self.dence_out(X)
        #         X = torch.sigmoid(X)
        X = self.log_soft_max(X)
        return X, hidden_state
