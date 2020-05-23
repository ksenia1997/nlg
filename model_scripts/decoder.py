import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, config, vocab, device):
        super().__init__()
        self.embedding_dim = config["embedding_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.output_dim = len(vocab)
        self.n_layers = config["num_layers"]
        self.dropout_rate = config['dropout_rate']
        self.device = device
        self.embedder = nn.Embedding(self.output_dim, self.embedding_dim).to(self.device)
        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            self.n_layers,
            dropout=float(self.dropout_rate)).to(self.device)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)
        self.dropout = nn.Dropout(self.dropout_rate).to(self.device)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden=None, cell=None):
        """

        Args:
            inputs: generating input by the decoder
            hidden: previous hidden state
            cell: precious cell

        Returns: generated output, hidden and cell

        """
        inputs = inputs.unsqueeze(0)
        # inputs [1,batch_size]
        embedded = self.embedder(inputs).to(self.device)
        embedded = self.dropout(embedded).to(self.device)
        # embedded [1, batch_size, embedded_dim]
        if hidden is None or cell is None:
            output, (hidden, cell) = self.lstm(embedded)
        else:
            output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.log_softmax(self.linear(output.squeeze(0)).to(self.device))
        return output, hidden, cell


class LuongDecoder(nn.Module):
    def __init__(self, config, vocab):
        super(LuongDecoder, self).__init__()
        self.hidden_dim = config["hidden_dim"]
        self.output_dim = len(vocab)
        self.n_layers = config["num_layers"]
        self.dropout = config["dropout_rate"]
        self.attention_model = Attention(self.hidden_dim, config["attention_model"])

        self.embedding = nn.Embedding(self.output_dim, self.hidden_dim)
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.n_layers)
        self.classifier = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def forward(self, input_seq, hidden, encoder_outputs):
        """

        Args:
            input_seq: generating input sequence
            hidden: previous hidden state
            encoder_outputs: encoder outputs

        Returns: generating output, hidden state and attention weights

        """
        # Embed input words
        embedded = self.embedding(input_seq).unsqueeze(0)
        embedded = self.embedding_dropout(embedded)
        # hidden = (hidden[0].detach(), hidden[1].detach())
        lstm_out, hidden = self.lstm(embedded, hidden)
        alignment_scores = self.attention_model(lstm_out, encoder_outputs)
        attn_weights = F.softmax(alignment_scores, dim=1)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        output = torch.cat((lstm_out, context_vector.permute(1, 0, 2)), -1)
        output = F.log_softmax(self.classifier(output[0]), dim=1)

        return output, hidden, attn_weights


class Attention(nn.Module):
    def __init__(self, hidden_size, method='dot'):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)

        if self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
            self.weight = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, decoder_hidden, encoder_outputs):
        """

        Args:
            decoder_hidden: hidden state from the decoder
            encoder_outputs: encoder outputs

        Returns: returned weighted sum depending on the chosen attention method

        """
        decoder_hidden = decoder_hidden.permute(1, 0, 2)
        if self.method == "dot":
            # For the dot scoring method, no weights or linear layers are involved
            return torch.sum(decoder_hidden * encoder_outputs, dim=2)

        elif self.method == "general":
            # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
            energy = self.attn(encoder_outputs)
            return torch.sum(decoder_hidden * energy, dim=2)

        elif self.method == "concat":
            energy = self.attn(
                torch.cat((decoder_hidden.expand(-1, encoder_outputs.size(1), -1), encoder_outputs), 2)).tanh()
            return torch.sum(self.weight * energy, dim=2)
