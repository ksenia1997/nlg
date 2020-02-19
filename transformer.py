import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtext import *
from torchtext.data import Field

from preprocessing import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model).to(device)

    def forward(self, x):
        return self.embed(x).to(device)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=1000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout).to(device)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model, device=device)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if device == 0:
        np_mask = np_mask.cuda()
    return np_mask


def create_masks(src, trg):
    src_mask = (src != TEXT.vocab.stoi['<pad>']).unsqueeze(-2)
    if trg is not None:
        trg_mask = (trg != TEXT.vocab.stoi['<pad>']).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size)
        trg_mask = trg_mask & np_mask.to(device)

    else:
        trg_mask = None
    return src_mask, trg_mask


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model).to(device)
        self.v_linear = nn.Linear(d_model, d_model).to(device)
        self.k_linear = nn.Linear(d_model, d_model).to(device)

        self.dropout = nn.Dropout(dropout).to(device)
        self.out = nn.Linear(d_model, d_model).to(device)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
                                           src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model).to(device)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size, device=device))
        self.bias = nn.Parameter(torch.zeros(self.size, device=device))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model).to(device)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout).to(device)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout).to(device)
        self.out = nn.Linear(d_model, trg_vocab).to(device)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        # print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


def train():
    total_loss = 0
    model.train()
    print("Train")
    for i, batch in enumerate(train_iter):
        src = batch.src.transpose(0, 1)
        trg = batch.trg.transpose(0, 1)
        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input)
        preds = model(src, trg_input, src_mask, trg_mask)
        ys = trg[:, 1:].contiguous().view(-1)
        optimizer.zero_grad()
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=TEXT.vocab.stoi['<pad>'])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_iter)


def test_model(text, max_len):
    model.eval()
    _, tokenized = tokenize(text, nlp)
    tokenized = [TEXT.vocab.stoi['<sos>']] + tokenized + [TEXT.vocab.stoi['<eos>']]
    numericalized = [TEXT.vocab.stoi[t] for t in tokenized]
    sentence = Variable(torch.LongTensor(numericalized).to(device))
    src_mask = (sentence != TEXT.vocab.stoi['<pad>']).unsqueeze(-2)
    e_output = model.encoder(sentence, src_mask)

    outputs = torch.LongTensor([[TEXT.vocab.stoi['<sos>']]])
    trg_mask = nopeak_mask(1)

    trg = []
    for i in range(1, max_len):
        print("Outputs: ", outputs.size())
        out = model.out(model.decoder(outputs[-1], e_output, src_mask, trg_mask))
        print("Out: ", out.size())
        out = F.softmax(out, dim=-1)
        print("Out softmax: ", out.size())
        probs, ix = out.topk(1)
        print("probs: ", probs.size())
        print("IX: ", ix.size())
        trg.append(probs)
        outputs = torch.cat((outputs, torch.LongTensor([probs]).to(device)), 0)

    return [TEXT.vocab.itos[k] for k in trg]
    # log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)


def evaluate():
    model.eval()
    total_loss = 0
    print("Evaluate")
    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)
        trg = batch.trg.transpose(0, 1)
        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input)
        preds = model(src, trg_input, src_mask, trg_mask)
        ys = trg[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=TEXT.vocab.stoi['<pad>'])
        total_loss += loss.item()
    return total_loss / len(train_iter)


def train_model(epochs):
    print("training model...")
    best_validation_loss = float('inf')

    for epoch in range(epochs):
        train_loss = train()
        valid_loss = evaluate()
        if valid_loss < best_validation_loss:
            best_validation_loss = valid_loss
            torch.save(model.state_dict(), "transformer.pt")
            print("Valid loss is less: ", valid_loss, train_loss)
        else:
            print("Valid loss NOT less: ", valid_loss, train_loss)


TEXT = Field(sequential=True, tokenize=lambda s: str.split(s, sep=' '), init_token='<sos>',
             eos_token='<eos>', pad_token='<pad>', lower=True)

data_fields = [('src', TEXT), ('trg', TEXT)]

trn, vld, test = data.TabularDataset.splits(
    path="./datasets",  # the root directory where the data lies
    train='train.csv', validation="valid.csv", test='test.csv',
    format='csv',
    skip_header=True,
    fields=data_fields)

TEXT.build_vocab(trn)
train_iter = data.BucketIterator(trn, shuffle=True, sort=False,
                                 batch_size=config["train_batch_size"],
                                 repeat=False,
                                 device=device)

valid_iter = data.BucketIterator(vld, shuffle=False, sort=False,
                                 batch_size=config["train_batch_size"],
                                 repeat=False,
                                 device=device)

model = Transformer(len(TEXT.vocab), len(TEXT.vocab), 512, 6, 8, 0.1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
print("Start training")
train_model(4)
#model.load_state_dict(torch.load("transformer.pt", map_location=torch.device(device)))
test_data = load_csv('datasets/test.csv')
data_to_save = []
for i in range(0, len(test_data), 2):
    answer = test_model(test_data[i], 20)
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
