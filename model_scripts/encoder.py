import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, config, vocab, device):
        super().__init__()

        self.embedding_dim = config["embedding_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.n_layers = config["num_layers"]
        self.dropout_rate = config['dropout_rate']
        self.vocab_size = len(vocab)
        self.device = device

        self.embedder = nn.Embedding(self.vocab_size, self.embedding_dim).to(device)
        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            self.n_layers,
            dropout=float(self.dropout_rate)).to(device)
        self.dropout = nn.Dropout(self.dropout_rate).to(device)

    def forward(self, input_sequence, pad_token=None):
        """

        Args:
            input_sequence: input sequence for the encoding
            pad_token: index of the PAD token

        Returns: encoded outputs, hidden state, cell

        """
        use_padded = False
        if isinstance(input_sequence, tuple):
            use_padded = True
            input_lengths = input_sequence[1]
            input_sequence = input_sequence[0]
        embeds_input = self.embedder(input_sequence).to(self.device)
        embedded = self.dropout(embeds_input).to(self.device)
        if use_padded:
            inp_packed = pack_padded_sequence(embedded, input_lengths, batch_first=False, enforce_sorted=False)
            outputs, (hidden, cell) = self.lstm(inp_packed)
            outputs, output_lengths = pad_packed_sequence(outputs, batch_first=False,
                                                          padding_value=pad_token,
                                                          total_length=input_sequence.shape[0])
        else:
            outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell
