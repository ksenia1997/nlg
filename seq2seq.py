import torch
import torch.nn as nn
import re
import en_core_web_sm
import spacy
import string
from spacy.tokenizer import Tokenizer
from torchtext.data import Field
import csv
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
from torchtext.vocab import GloVe
import random
import torch.optim as optim
import math
import torch.nn.functional as F
import operator
from queue import PriorityQueue

SEED = 5  # set seed value for deterministic results
N_EPOCHS = 15
CLIP = 10
JOIN_TOKEN = " "
TEST_QUESTION = "Hi, how are you?"
CONTEXT_PAIR_COUNT = 0
IS_TEST = False
DEBUG = False
WITH_DESCRIPTION = True
# Create Field object
# TEXT = data.Field(tokenize = 'spacy', lower=True, include_lengths = True, init_token = '<sos>',  eos_token = '<eos>')
TEXT = Field(sequential=True, tokenize=lambda s: str.split(s, sep=JOIN_TOKEN), init_token='<sos>', eos_token='<eos>',
             pad_token='<pad>', lower=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = 'seq2seq_model.pt'
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def read_text(filename, context_pair_count):
    print("Reading training dataset from Stanford")
    lines = []
    counter = 0
    your_persona_description = ""
    with open(filename) as fp:
        question_line = ""
        for line in fp:
            if line == '\n':
                question_line = ""
            your_persona = re.findall(r"(your persona:.*\\n)", line)
            if WITH_DESCRIPTION and len(your_persona) > 0:
                your_persona = re.sub(r"\\n", '', your_persona[0]).split("your persona: ")
                your_persona_description = ' # '.join(your_persona[1:])
                question_line += your_persona_description
            line = re.sub(r"(your persona:.*\\n)", ' ', line)
            line = ' '.join(line.split())
            question = re.findall(r"text:(.*)labels:", line)
            answer = re.findall(r"labels:(.*)episode_done:", line)
            if len(answer) == 0:
                answer = re.findall(r"labels:(.*)question:", line)
            if len(answer) and len(question):
                if counter < context_pair_count or context_pair_count == 0:
                    question_line += " # " + question[0]
                    counter += 1
                else:
                    question_line = your_persona_description + " # " + question[0]
                    counter = 0
                answer_line = question_line + " # " + answer[0]
                lines.append(question_line)
                lines.append(answer_line)

    return lines


def tokenize_and_join(text: string, t, jointoken=JOIN_TOKEN):
    return jointoken.join(tokenize(text, t)[1])


def create_data(name, lines, from_line, to_line):
    with open(name, mode='w') as csv_file:
        fieldnames = ['question', 'answer']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(from_line, to_line, 2):
            writer.writerow({'question': lines[i], 'answer': lines[i + 1]})


def prepare_data():
    print("Prepare data")
    lines = read_text('train.txt', CONTEXT_PAIR_COUNT)
    tokenized_lines = []
    for line in lines:
        tokenized_lines.append(tokenize_and_join(line, nlp))

    lines = tokenized_lines
    create_data('train.csv', lines, 0, int(len(lines) * 9 / 10))
    create_data('valid.csv', lines, int(len(lines) * 9 / 10), len(lines))
    create_data('test.csv', lines, int(len(lines) / 8), int(len(lines) * 3 / 8))


def create_custom_tokenizer(nlp):
    print("Creating custom tokenizer")
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


def tokenize(text: string, t):
    tokens = [tok for tok in t.tokenizer(text) if not tok.text.isspace()]
    text_tokens = [tok.text for tok in tokens]
    return tokens, text_tokens


class Attention(nn.Module):
    # Global Attention model described in LINK(Luong et. al)
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size

    def dot_score(self, hidden_state, encoder_states):
        return torch.sum(hidden_state * encoder_states, dim=2)

    def forward(self, hidden, encoder_outputs, mask):
        attn_scores = self.dot_score(hidden, encoder_outputs)
        # Transpose max_length and batch_size dimensions
        attn_scores = attn_scores.t()
        # Apply mask so network does not attend <pad> tokens
        attn_scores = attn_scores.masked_fill(mask == 0, -1e10)
        # Return softmax over attention scores
        return F.softmax(attn_scores, dim=1).unsqueeze(1)


def greedy_search(model, vocab, fields, trg_indexes, hidden, cell, max_len):
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        predicted, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = predicted.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == vocab.stoi[fields['answer'].eos_token]:
            break
        return [vocab.itos[i] for i in trg_indexes]


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def beam_decode(decoder, vocab, fields, target_tensor, decoder_hiddens, encoder_output):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''
    print("Beam decode")
    beam_width = 20
    topk = 3  # how many sentence do you want to generate
    decoded_batch = []
    EOS_token = fields['answer'].eos_token
    SOS_token = fields['answer'].init_token

    # decoding goes sentence by sentence
    for idx in range(target_tensor.size(0)):
        decoder_hidden = decoder_hiddens

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([vocab.stoi[SOS_token]]).to(device)
        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 1000: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == EOS_token and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden, cell = decoder(decoder_input, decoder_hidden, encoder_output)
            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].unsqueeze(0)
                log_p = log_prob[0][new_k].item()
                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)
            utterance = utterance[::-1]
            sentence = [vocab.itos[i] for i in utterance]
            decoded_batch.append(sentence)

    return decoded_batch


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


class Encoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.embedding_dim = config["embedding_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.n_layers = config["num_layers"]
        self.dropout_rate = config['dropout_rate']

        self.embedder = nn.Embedding(len(vocab), self.embedding_dim)
        self.lstm = torch.nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            self.n_layers,
            dropout=float(self.dropout_rate))
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, input_sequence):
        # Convert input_sequence to word embeddings
        embeds_q = self.embedder(input_sequence)
        enc_q = self.dropout(embeds_q)
        outputs, (hidden, cell) = self.lstm(enc_q)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.embedding_dim = config["embedding_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.output_dim = len(vocab)
        self.n_layers = config["num_layers"]
        self.dropout_rate = config['dropout_rate']

        self.embedder = nn.Embedding(self.output_dim, self.embedding_dim)
        self.lstm = torch.nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            self.n_layers,
            dropout=float(self.dropout_rate))
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

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
        print("trg size: ", trg.size())
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
    print("Train")
    model.train()
    # loss
    epoch_loss = 0

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
    print("Evaluate")
    #  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
    model.eval()
    # loss
    epoch_loss = 0

    # we don't need to update the model parameters. only forward pass.
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.question
            trg = batch.answer

            output = model(src, trg, 0)  # turn off the teacher forcing
            # loss function works only 2d logits, 1d targets
            # so flatten the trg, output tensors. Ignore the <sos> token
            # trg shape shape should be [(sequence_len - 1) * batch_size]
            # output shape should be [(sequence_len - 1) * batch_size, output_dim]
            loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))

            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def test_model(example, fields, vocab, model, max_len=10):
    print("Test model")
    model.eval()
    _, tokenized = tokenize(example, nlp)
    tokenized = [fields['question'].init_token] + tokenized + [fields['question'].eos_token]
    numericalized = [vocab.stoi[t] for t in tokenized]
    src_tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)
    hidden, cell = model.encoder(src_tensor)
    trg_indexes = [vocab.stoi[fields['answer'].init_token]]
    trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
    trg_tensor = beam_decode(model.decoder, vocab, fields, trg_tensor, hidden, cell)

    return trg_tensor


def train_model(model, fields, train_iter, valid_iter):
    print("Train model")
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters())
    pad_idx = fields['question'].vocab.stoi[fields['answer'].pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    best_validation_loss = float('inf')

    for epoch in range(N_EPOCHS):
        print("epoch: ", epoch)
        train_loss = train(model, train_iter, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iter, criterion)
        if valid_loss < best_validation_loss:
            best_validation_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(
                f'| Epoch: {epoch + 1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')


def main():
    if DEBUG:
        prepare_data()
        exit()
    config = {"train_batch_size": 80, "optimize_embeddings": False,
              "embedding_dim": 100, "hidden_dim": 200, "dropout_rate": 0.5, "num_layers": 2}

    # Specify Fields in our dataset
    data_fields = [('question', TEXT), ('answer', TEXT)]
    fields = dict(data_fields)

    # Build the dataset for train, validation and test sets
    trn, vld, test = TabularDataset.splits(
        path="~/nlg",  # the root directory where the data lies
        train='train.csv', validation="valid.csv", test='test.csv',
        format='csv',
        skip_header=True,
        # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
        fields=data_fields)

    # Build vocabulary
    print("Build vocabulary")
    fields["question"].build_vocab(trn, vectors=GloVe(name='6B', dim=config["embedding_dim"]))
    vocab = fields["question"].vocab

    # Create a set of iterators
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

    print("Most common: ", vocab.freqs.most_common(50))

    enc = Encoder(config, vocab)
    dec = Decoder(config, vocab)
    model = Seq2Seq(enc, dec)

    if IS_TEST:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        answer = test_model(TEST_QUESTION, fields, vocab, model)
        print("QUESTION: ", TEST_QUESTION)
        print("ANSWER: ", answer)
    else:
        train_model(model, fields, train_iter, valid_iter)


if __name__ == "__main__":
    main()
