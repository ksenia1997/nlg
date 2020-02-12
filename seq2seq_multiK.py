import random
import time

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import Field, BucketIterator
from torchtext.data import TabularDataset
from torchtext.vocab import GloVe

from params import *
from preprocessing import tokenize, nlp
from utils.csv import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


SRC = Field(tokenize=lambda s: str.split(s, sep=JOIN_TOKEN),
            include_lengths=True,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

TRG = Field(tokenize=lambda s: str.split(s, sep=JOIN_TOKEN),
            include_lengths=True,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

train_data, valid_data, test_data = TabularDataset.splits(
    path="./datasets",  # the root directory where the data lies
    train='train.csv', validation="valid.csv", test='test.csv',
    format='csv',
    skip_header=False,
    fields=[('question', SRC), ('answer', TRG)])

print(vars(train_data.examples[0]))

ENC_EMB_DIM = 300
DEC_EMB_DIM = 300
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

SRC.build_vocab(train_data, vectors=GloVe(name='6B', dim=ENC_EMB_DIM))
TRG.build_vocab(train_data, vectors=GloVe(name='6B', dim=DEC_EMB_DIM))

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    sort=False,
    batch_size=BATCH_SIZE,
    device=device)


class Encoder(nn.Module):
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
        use_padded = False
        if isinstance(input_sequence, tuple):
            use_padded = True
            input_lengths = input_sequence[1]
            input_sequence = input_sequence[0]
        embeds_q = self.embedder(input_sequence).to(device)
        embedded = self.dropout(embeds_q).to(device)
        if use_padded:
            inp_packed = pack_padded_sequence(embedded, input_lengths, batch_first=False, enforce_sorted=False)
            outputs, (hidden, cell) = self.lstm(inp_packed)
            outputs, output_lengths = pad_packed_sequence(outputs, batch_first=False,
                                                          padding_value=input_sequence[0][0],
                                                          total_length=input_sequence.shape[0])
        else:
            outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Decoder(nn.Module):
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


class Seq2Seq(nn.Module):
    def __init__(self, config, vocab_src, vocab_trg):
        super().__init__()
        self.encoder = Encoder(config, vocab_src).to(device)
        self.decoder = Decoder(config, vocab_trg).to(device)
        experiment_name = "train_" + time.strftime('%d-%m-%Y_%H:%M:%S')
        tensorboard_log_dir = './tensorboard-logs/{}/'.format(experiment_name)
        self.tb = SummaryWriter(tensorboard_log_dir)

        assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src [seq_len, batch_size]
        # trg [seq_len, batch_size]
        max_len, batch_size = trg[0].size()
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(device)
        enc_output, hidden, cell = self.encoder(src)
        decoder_input = trg[0][0, :]
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[t] = output
            use_teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = trg[0][t] if use_teacher_force else top1

        return outputs.to(device)


model = Seq2Seq(config, SRC.vocab, TRG.vocab).to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.question
        trg = batch.answer

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[0][1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.question
            trg = batch.answer

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[:-1].view(-1, output_dim)
            trg = trg[0][1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def greedy_decode(model, vocab, trg_indexes, hidden, cell, max_len):
    trg = []
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        predicted, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = predicted.argmax(1).item()
        trg.append(pred_token)
        trg_indexes = torch.cat((trg_indexes, torch.LongTensor([pred_token]).to(device)), 0)
        if pred_token == vocab.stoi[TRG.eos_token]:
            break
    return [vocab.itos[i] for i in trg]


def test(example, vocab, model):
    model.eval()
    _, tokenized = tokenize(example, nlp)
    tokenized = [SRC.init_token] + tokenized + [SRC.eos_token]
    numericalized = [vocab.stoi[t] for t in tokenized]
    src_tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)
    output, hidden, cell = model.encoder(src_tensor)
    trg_indexes = [vocab.stoi[TRG.init_token]]
    trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
    trg_tensor = greedy_decode(model, vocab, trg_tensor, hidden, cell, 10)
    return trg_tensor


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10

best_valid_loss = float('inf')

# for epoch in range(N_EPOCHS):
#
#     start_time = time.time()
#
#     train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
#     valid_loss = evaluate(model, valid_iterator, criterion)
#
#     end_time = time.time()
#
#     model.tb.add_scalar('train_loss', train_loss, epoch)
#     model.tb.add_scalar('valid_loss', valid_loss, epoch)
#     for name, param in model.named_parameters():
#         # print("name: ", name, param)
#         if param.grad is not None and not param.grad.data.is_sparse:
#             # print("param grad: ", param.grad)
#             # print("data: ", param.grad.data)
#             model.tb.add_histogram(f"gradients_wrt_hidden_{name}/",
#                                    param.grad.data.norm(p=2, dim=0),
#                                    global_step=epoch)
#
#     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
#
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), 'tut1-model.pt')
#
#     print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
#     print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
#     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load('tut1-model.pt', map_location=torch.device(device)))
test_data = load_csv('datasets/test.csv')
data_to_save = []
for i in range(0, len(test_data), 2):
    answer = test(test_data[i], TRG.vocab, model)
    answer_str = ""
    for a in answer:
        answer_str += a + " "
    data_to_save.append(test_data[i])
    data_to_save.append(answer_str)
    if i % 1000 == 0:
        print("QUESTION: ", test_data[i])
        print("ANSWER: ", answer_str)
file_path = "./tests/" + time.strftime('%d-%m-%Y_%H:%M:%S') + ".csv"
save_to_csv(file_path, data_to_save)
