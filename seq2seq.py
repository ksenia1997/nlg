import numpy as np
import torch
import torch.nn as nn
import re
import unicodedata
import spacy
import string
from spacy.tokenizer import Tokenizer
from torchtext.data import Field
import en_core_web_sm
import csv
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
from torchtext.vocab import GloVe
import random
import torch.optim as optim
import math

SEED = 5
JOIN_TOKEN = " "
TEXT = Field(sequential=True, tokenize=lambda s: str.split(s, sep=JOIN_TOKEN), init_token='<sos>', eos_token='<eos>',
             lower=True)
N_EPOCHS = 10
CLIP = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_text():
    print("Reading training dataset from Stanford")
    lines = []
    with open('train.txt') as fp:
        for line in fp:
            line = re.sub(r"(your persona:.*\\n)", ' ', line)
            line = ' '.join(line.split())
            question = re.findall(r"text:(.*)labels:", line)
            answer = re.findall(r"labels:(.*)(episode_done:)", line)
            if len(answer) == 0:
                answer = re.findall(r"labels:(.*)(question:)", line)
            if len(answer) and len(question):
                lines.append(question[0])
                lines.append(answer[0][0])

    return lines


def create_custom_tokenizer(nlp):
    custom_prefixes = [r'[0-9]+', r'\~', r'\–', r'\—', r'\$']
    custom_infixes = [r'[!&:,()]', r'\.', r'\-', r'\–', r'\—', r'\$']
    custom_suffixes = [r'\.', r'\–', r'\—', r'\$']
    default_prefixes = list(nlp.Defaults.prefixes) + custom_prefixes
    default_prefixes.remove(r'US\$')
    default_prefixes.remove(r'C\$')
    default_prefixes.remove(r'A\$')

    all_prefixes_re = spacy.util.compile_prefix_regex(tuple(default_prefixes))
    infix_re = spacy.util.compile_infix_regex(tuple(list(nlp.Defaults.infixes) + custom_infixes))
    suffix_re = spacy.util.compile_suffix_regex(tuple(list(nlp.Defaults.suffixes) + custom_suffixes))

    rules = dict(nlp.Defaults.tokenizer_exceptions)
    # remove "a." to "z." rules so "a." gets tokenized as a|.
    for c in range(ord("a"), ord("z") + 1):
        if f"{chr(c)}." in rules:
            rules.pop(f"{chr(c)}.")

    return Tokenizer(nlp.vocab, rules,
                     prefix_search=all_prefixes_re.search,
                     infix_finditer=infix_re.finditer, suffix_search=suffix_re.search,
                     token_match=None)


nlp = en_core_web_sm.load()
nlp.tokenizer = create_custom_tokenizer(nlp)


def tokenize(text: string, tokenizer=nlp):
    tokens = [tok for tok in nlp.tokenizer(text) if not tok.text.isspace()]
    text_tokens = [tok.text for tok in tokens]
    return tokens, text_tokens


def tokenize_and_join(text: string, jointoken=JOIN_TOKEN):
    return jointoken.join(tokenize(text)[1])


def create_data(name, lines, from_line, to_line):
    with open(name, mode='w') as csv_file:
        fieldnames = ['question', 'answer']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(from_line, to_line, 2):
            writer.writerow({'question': lines[i], 'answer': lines[i + 1]})


class Embedder(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.vocab = vocab
        self.init_vocab(vocab, config['optimize_embeddings'])
        print(f"Optimize embeddings = {config['optimize_embeddings']}")
        print(f"Vocabulary size = {len(vocab.vectors)}")

    def init_vocab(self, vocab, optimize_embeddings=False, device=None):
        self.embeddings = nn.Embedding(len(vocab), vocab.vectors.shape[1])

        if device is not None:
            self.embeddings = self.embeddings.to(device)
        # Copy over the pre-trained GloVe embeddings
        self.embeddings.weight.data.copy_(vocab.vectors)
        self.embeddings.weight.requires_grad = optimize_embeddings

    def forward(self, input):
        return self.embeddings(input)


class Encoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.embedder = nn.Embedding(len(vocab), config["embedding_dim"])

        self.hidden_dim = config["hidden_dim"]
        self.n_layers = config["num_layers"]

        self.lstm = torch.nn.LSTM(
            config["embedding_dim"],
            config["hidden_dim"],
            config["num_layers"],
            dropout=float(config['dropout_rate']))

        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, batch):
        embeds_q = self.embedder(batch)
        enc_q = self.dropout(embeds_q)
        outputs, (hidden, cell) = self.lstm(enc_q)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.output_dim = len(vocab)
        self.embedder = nn.Embedding(self.output_dim, config["embedding_dim"])
        self.lstm = torch.nn.LSTM(
            config["embedding_dim"],
            config["hidden_dim"],
            config["num_layers"],
            dropout=float(config['dropout_rate']))
        self.linear = nn.Linear(config["hidden_dim"], self.output_dim)
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.embedder(input)
        embedded = self.dropout(embedded)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        predicted = self.linear(output.squeeze(0))
        return predicted, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size)
        hidden, cell = self.encoder(src)
        input = trg[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            use_teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if use_teacher_force else top1)

        return outputs


def train(model, iterator, optimizer, criterion, clip):
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
    #  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
    print("train")
    model.train()
    # loss
    epoch_loss = 0
    print("train iterator")
    for i, batch in enumerate(iterator):
        src = batch.question
        trg = batch.answer
        optimizer.zero_grad()

        # trg is of shape [sequence_len, batch_size]
        # output is of shape [sequence_len, batch_size, output_dim]
        output = model(src, trg)

        # loss function works only 2d logits, 1d targets
        # so flatten the trg, output tensors. Ignore the <sos> token
        # trg shape shape should be [(sequence_len - 1) * batch_size]
        # output shape should be [(sequence_len - 1) * batch_size, output_dim]
        loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))

        # backward pass
        loss.backward()

        # clip the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # update the parameters
        optimizer.step()

        epoch_loss += loss.item()

    # return the average loss
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    ''' Evaluation loop for the model to evaluate.
    Args:
        model: A Seq2Seq model instance.
        iterator: A DataIterator to read the data.
        criterion: loss criterion.
    Returns:
        epoch_loss: Average loss of the epoch.
    '''
    #  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
    model.eval()
    # loss
    epoch_loss = 0

    # we don't need to update the model parameters. only forward pass.
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off the teacher forcing

            # loss function works only 2d logits, 1d targets
            # so flatten the trg, output tensors. Ignore the <sos> token
            # trg shape shape should be [(sequence_len - 1) * batch_size]
            # output shape should be [(sequence_len - 1) * batch_size, output_dim]
            loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))

            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # We will use this special token to join the pre-tokenized data
    # lines = read_text()
    # tokenized_lines = []
    # for line in lines:
    #     tokenized_lines.append(tokenize_and_join(line))
    #
    # print(tokenize_and_join(lines[0]))
    # lines = tokenized_lines
    # create_data('train.csv', lines, 0, int(len(lines) * 2 / 3))
    # create_data('valid.csv', lines, int(len(lines) * 2 / 3), len(lines))
    # create_data('test.csv', lines, int(len(lines) / 5), int(len(lines) / 5 * 4))

    config = {"train_batch_size": 80, "optimize_embeddings": False,
              "scale_emb_grad_by_freq": False, "embedding_dim": 100, "hidden_dim": 200, "dropout_rate": 0.5,
              "num_layers": 2}

    data_fields = [
        ('question', TEXT),
        ('answer', TEXT)
    ]
    fields = dict(data_fields)

    # train = SDataset(fields)
    trn, vld, test = TabularDataset.splits(
        path="~/nlg",  # the root directory where the data lies
        train='train.csv', validation="valid.csv", test='test.csv',
        format='csv',
        skip_header=True,
        # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
        fields=data_fields)

    fields["question"].build_vocab(trn, vectors=GloVe(name='6B', dim=config["embedding_dim"]))
    vocab = fields["question"].vocab

    train_iter = BucketIterator(trn,
                                shuffle=True, sort=False,
                                batch_size=config["train_batch_size"],
                                repeat=False,
                                device=device)
    valid_iter = BucketIterator(vld,
                                shuffle=True, sort=False,
                                batch_size=config["train_batch_size"],
                                repeat=False,
                                device=device)

    print(vocab.freqs.most_common(50))
    enc = Encoder(config, vocab)
    dec = Decoder(config, vocab)
    model = Seq2Seq(enc, dec)
    optimizer = optim.Adam(model.parameters())
    pad_idx = fields['question'].vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    best_validation_loss = float('inf')

    for epoch in range(N_EPOCHS):
        print("epoch: ", epoch)
        train_loss = train(model, train_iter, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iter, criterion)

        if valid_loss < best_validation_loss:
            best_validation_loss = valid_loss
            # torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(
                f'| Epoch: {epoch + 1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')


if __name__ == "__main__":
    main()
