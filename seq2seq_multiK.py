import numpy as np
from torchtext.datasets import Multi30k

from seq2seq import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def training(model, iterator, optimizer, criterion):
    ''' Training loop for the model to train.
    Args:
        model: A Seq2Seq model instance.
        iterator: A DataIterator to read the data.
        optimizer: Optimizer for the model.
        criterion: loss criterion.
        clip: gradient clip value.
    Returns:
        epoch_loss: Average loss of the epoch.
    '''
    print("Train multi K")
    #  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
    model.train()
    # loss
    epoch_loss = 0
    print("iterator: ", len(iterator))
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        # trg is of shape [sequence_len, batch_size]
        # output is of shape [sequence_len, batch_size, output_dim]
        output = model(src, trg, float('inf'))

        # trg shape shape should be [(sequence_len - 1) * batch_size]
        # output shape should be [(sequence_len - 1) * batch_size, output_dim]
        loss = criterion(output[:-1].view(-1, output.shape[2]), trg[0][1:].view(-1))
        # backward pass
        loss.backward()

        # clip the gradients
        clip_grad_norm_(model.parameters(), CLIP)

        # update the parameters
        optimizer.step()

        epoch_loss += loss.item()

    # return the average loss
    return epoch_loss / len(iterator)


def evaluating(model, iterator, criterion):
    ''' Evaluation loop for the model to evaluate.
    Args:
        model: A Seq2Seq model instance.
        iterator: A DataIterator to read the data.
        criterion: loss criterion.
    Returns:
        epoch_loss: Average loss of the epoch.
    '''

    model.eval()
    epoch_loss = 0
    print("Evaluate multi K")
    # we don't need to update the model parameters. only forward pass.
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, 0)  # turn off the teacher forcing
            # trg shape shape should be [(sequence_len - 1) * batch_size]
            # output shape should be [(sequence_len - 1) * batch_size, output_dim]
            loss = criterion(output[:-1].view(-1, output.shape[2]), trg[0][1:].view(-1))
            epoch_loss += loss.item()
            # if (i + 1) % 100 == 0:
            #    print("eval loss: ", epoch_loss / i)
    return epoch_loss / len(iterator)


class EncoderM(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.embedding_dim = config["embedding_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.n_layers = config["num_layers"]
        self.dropout_rate = config['dropout_rate']
        self.vocab_size = len(vocab)
        self.embedder = nn.Embedding(self.vocab_size, self.embedding_dim).to(device)
        self.lstm = torch.nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            self.n_layers,
            dropout=float(self.dropout_rate)).to(device)
        self.dropout = nn.Dropout(self.dropout_rate).to(device)

    def forward(self, input_sequence):
        # Convert input_sequence to word embeddings
        embeds_q = self.embedder(input_sequence[0]).to(device)
        embedded = self.dropout(embeds_q).to(device)
        inp_packed = pack_padded_sequence(embedded, input_sequence[1], batch_first=False, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(inp_packed)
        outputs, output_lengths = pad_packed_sequence(outputs, batch_first=False,
                                                      padding_value=input_sequence[0][0][0],
                                                      total_length=input_sequence[0].shape[0])
        return outputs, hidden, cell


class DecoderM(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.embedding_dim = config["embedding_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.output_dim = len(vocab)
        self.n_layers = config["num_layers"]
        self.dropout_rate = config['dropout_rate']
        self.embedder = nn.Embedding(self.output_dim, self.embedding_dim).to(device)
        self.lstm = torch.nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            self.n_layers,
            dropout=float(self.dropout_rate)).to(device)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim).to(device)
        self.dropout = nn.Dropout(self.dropout_rate).to(device)

    def forward(self, inputs, hidden, cell):
        inputs = inputs.unsqueeze(0)
        # inputs [1,batch_size]
        embedded = self.embedder(inputs).to(device)
        embedded = self.dropout(embedded).to(device)
        # embedded [1, batch_size, embedded_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predicted = self.linear(output.squeeze(0)).to(device)
        return predicted, hidden, cell


class Seq2SeqM(nn.Module):
    def __init__(self, config, vocab_src, vocab_trg):
        super().__init__()
        self.encoder = EncoderM(config, vocab_src).to(device)
        if WITH_ATTENTION:
            self.decoder = LuongDecoder(config, vocab_trg).to(device)
        else:
            self.decoder = DecoderM(config, vocab_trg).to(device)

        assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

        experiment_name = "train_" + time.strftime('%d-%m-%Y_%H:%M:%S')
        tensorboard_log_dir = './tensorboard-logs/{}/'.format(experiment_name)
        self.tb = SummaryWriter(tensorboard_log_dir)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src [seq_len, batch_size]
        # trg [seq_len, batch_size]
        max_len, batch_size = trg[0].size()
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(device)
        enc_output, hidden, cell = self.encoder(src)
        enc_output = enc_output.permute(1, 0, 2)
        decoder_h = (hidden, cell)
        decoder_input = trg[0][0, :]
        for t in range(1, max_len):
            if WITH_ATTENTION:
                output, decoder_h, attn_weights = self.decoder(decoder_input, decoder_h, enc_output)
            else:
                output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[t] = output
            use_teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            decoder_input = trg[0][t] if use_teacher_force else top1

        return outputs.to(device)


def model_fit(model, train_iter, valid_iter):
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss(ignore_index=SRC_PAD_IDX)
    best_validation_loss = float('inf')

    for epoch in range(N_EPOCHS):
        train_loss = training(model, train_iter, optimizer, criterion)
        valid_loss = evaluating(model, valid_iter, criterion)
        model.tb.add_scalar('train_loss', train_loss, epoch)
        model.tb.add_scalar('valid_loss', valid_loss, epoch)
        for name, param in model.named_parameters():
            # print("name: ", name, param)
            if param.grad is not None and not param.grad.data.is_sparse:
                # print("param grad: ", param.grad)
                # print("data: ", param.grad.data)
                model.tb.add_histogram(f"gradients_wrt_hidden_{name}/",
                                       param.grad.data.norm(p=2, dim=0),
                                       global_step=epoch)
        if valid_loss < best_validation_loss:
            best_validation_loss = valid_loss
            torch.save(model.state_dict(), "multiK.pt")
            print(
                f'| Epoch: {epoch + 1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')
    model.tb.close()


SRC = Field(tokenize=tokenize_de,
            include_lengths=True,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

TRG = Field(tokenize=tokenize_en,
            include_lengths=True,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                    fields=(SRC, TRG))

SRC.build_vocab(train_data, vectors=GloVe(name='6B', dim=config["embedding_dim"]))
TRG.build_vocab(train_data, vectors=GloVe(name='6B', dim=config["embedding_dim"]))

vocab_src = SRC.vocab
vocab_trg = TRG.vocab
print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
model = Seq2SeqM(config, vocab_src, vocab_trg)
model_fit(model, train_iterator, valid_iterator)
