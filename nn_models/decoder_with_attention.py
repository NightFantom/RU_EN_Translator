import torch
from torch import nn
import torch.nn.functional as F

from gpu_utils.gpu_utils import get_device

BATCH_POS = 0
SEQ_LEN_POS = 1
FEATURE_POS = 2
HIDDEN_STATE_POS = 0
HIDDEN_FEATURE_POS = 1

class AttentionDecoder(nn.Module):

    def __init__(self, input_size, hidden_vector_size, vocabular_size, device):
        super().__init__()
        self.dence_in = nn.Linear(vocabular_size, input_size)
        self.decoder = nn.LSTM(batch_first=True, input_size= hidden_vector_size, hidden_size=hidden_vector_size)
        self.dence_out = nn.Linear(hidden_vector_size, vocabular_size)
        self.log_soft_max = nn.LogSoftmax(dim=-1)

        self.attention = nn.Linear(2 * hidden_vector_size, 1)

        self.dence_lstm =  nn.Linear(input_size + hidden_vector_size, hidden_vector_size)
        self.device = device

    def forward(self, X, hidden_state, encoder_output):
        """
        :param X: shape[batch x 1 x vocabular_size]
        :param hidden_state: shape[batch] - tuple. On the first position hidden state, on the second one cell state
            hidden_states[0].shape (1, batch, hidden_size)
            hidden_states[1].shape (1, batch, hidden_size)
        :param encoder_output: shape[batch x seq_len x hidden_size]
        :return:
        """
        transformed_X = self.dence_in(X)
        transformed_X = torch.tanh(transformed_X)

        context_matrix = self.get_context(encoder_output, hidden_state[HIDDEN_STATE_POS])

        dence_lstm_input = torch.cat((transformed_X, context_matrix), dim=2)
        lstm_input = self.dence_lstm(dence_lstm_input)
        lstm_input = F.relu(lstm_input)
        lstm_output, hidden_state = self.decoder(lstm_input, hidden_state)
        dence_output = self.dence_out(lstm_output)
        dence_output = self.log_soft_max(dence_output)
        return dence_output, hidden_state

    def get_context(self, encoder_output, hidden_state):
        """
        Get regularization parameters for vectorized by encoder words
        :param hidden_state: shape[1 x batch x hidden_size]
        :param encoder_output: shape[batch x seq_len x hidden_size]
        :return: context_matrix: shape[batch x 1 x hidden_size]
        """
        encoder_batch_size = encoder_output.shape[BATCH_POS]
        hidden_state = hidden_state[0]
        encoder_seq_len_int = encoder_output.shape[SEQ_LEN_POS]
        hidden_vector_size = hidden_state.shape[HIDDEN_FEATURE_POS]
        shape = (encoder_batch_size, 1, hidden_vector_size)
        context_matrix = torch.zeros(shape, device=self.device, dtype=torch.float32)
        for sentence_index in range(encoder_batch_size):
            encoded_sentence_torch = encoder_output[sentence_index]
            hidden_state_per_sentence = hidden_state[sentence_index]
            hidden_state_per_sentence = [hidden_state_per_sentence] * encoder_seq_len_int
            hidden_state_per_sentence = torch.stack(hidden_state_per_sentence, dim=0)

            # attention_input - shape[seq_len x 2 * hidden_size]
            attention_input = torch.cat((encoded_sentence_torch, hidden_state_per_sentence), 1)

            weights_torch = self.get_attention_coeficients(attention_input)
            context = weights_torch * encoded_sentence_torch
            # context: shape [hidden_size]
            context = torch.sum(context, dim=0)
            context_matrix[sentence_index][0] = context

        return context_matrix

    def get_attention_coeficients(self, attention_input):
        """

        :param attention_input: shape[seq_len x 2 * hidden_size]
        :return: shape[seq_len]
        """
        weights_torch = self.attention(attention_input)
        weights_torch = F.relu(weights_torch)
        # weights_torch: shape[seq_len]
        weights_torch = F.softmax(weights_torch, dim=0)
        return weights_torch


if __name__ == "__main__":
    input_size = 300
    hidden_vector_size  = 300
    vocabular_size = 500
    seq_len = 5
    device = get_device()
    decoder = AttentionDecoder(input_size, hidden_vector_size, vocabular_size, device)

    batch_size = 3
    X = torch.rand((batch_size, 1, vocabular_size), device=device, dtype=torch.float32)
    hidden_state = torch.rand((1, batch_size, hidden_vector_size), device=device, dtype=torch.float32)
    cell_state = torch.rand((1, batch_size, hidden_vector_size), device=device, dtype=torch.float32)
    encoder_output = torch.rand((batch_size, seq_len, hidden_vector_size), device=device, dtype=torch.float32)

    decoder(X, (hidden_state,cell_state), encoder_output)
