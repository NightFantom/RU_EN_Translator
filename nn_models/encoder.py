from torch import nn


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_vector_size):
        super().__init__()
        self.encoder = nn.LSTM(batch_first=True, input_size=input_size, hidden_size=hidden_vector_size)

    def forward(self, X):
        """
        X.shape [batch_size, seq_len, fast_text_vect]
        return: output.shape [batch_size, seq_len, hidden_size]
                hidden_states - typle
                    hidden_states[0].shape (1, batch_size, hidden_size)
                    hidden_states[1].shape (1, batch_size, hidden_size)
        """
        output, hidden_states = self.encoder(X)

        return output, hidden_states
