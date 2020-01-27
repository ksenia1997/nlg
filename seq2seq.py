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
import time
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter

SEED = 5  # set seed value for deterministic results
N_EPOCHS = 20
CLIP = 10
CONTEXT_PAIR_COUNT = 0
JOIN_TOKEN = " "

TEST_QUESTION = "Hi, how are you?"
# Preprocess
DATA_TYPE = "PERSONA"  # TWITTER or PERSONA
WITH_DESCRIPTION = True

WITH_ATTENTION = False
IS_BEAM_SEARCH = False

IS_TEST = True
DEBUG = False

# model_embeddingDim_hiddenDim_dropoutRate_numLayers_Epochs_batchSize
MODEL_SAVE_PATH = 'seq2seq_model.pt'

# Create Field object
# TEXT = data.Field(tokenize = 'spacy', lower=True, include_lengths = True, init_token = '<sos>',  eos_token = '<eos>')
TEXT = Field(sequential=True, tokenize=lambda s: str.split(s, sep=JOIN_TOKEN), init_token='<sos>',
             eos_token='<eos>', pad_token='<pad>', lower=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def save_to_csv(name, lines):
    with open(name, mode='w') as csv_file:
        fieldnames = ['question', 'answer']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        if len(lines) % 2 != 0:
            lines = lines[:-1]
        for i in range(0, len(lines), 2):
            writer.writerow({'question': lines[i], 'answer': lines[i + 1]})


def load_csv(name):
    with open(name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        lines = []
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                lines.append(row[0])
                lines.append(row[1])
                line_count += 1
    return lines


def prepare_Twitter_data(filename):
    train_data = []
    valid_data = []
    test_data = []
    counter = 0
    with open(filename) as fp:
        for line in fp:
            train_data.append(line)
            if counter % 10 == 0:
                valid_data.append(line)
            if counter % 20 == 0:
                test_data.append(line)
            counter += 1

    return train_data, valid_data, test_data


def prepare_Persona_chat(filename, context_pair_count):
    print("Reading Persona chat")
    train_data = []
    test_data = []
    valid_data = []
    context_pair_counter = 0
    line_counter = 0

    your_persona_description = ""
    add_to_test_data = True
    add_to_valid_data = True
    with open(filename) as fp:
        question_line = ""
        for line in fp:
            line_counter += 1
            if line == '\n':
                question_line = ""
                if random.randint(0, 100) < 5:
                    add_to_test_data = True
                    test_data.append("\n")
                    test_data.append("\n")
                else:
                    add_to_test_data = False
                if line_counter % 5 == 0:
                    add_to_valid_data = True
                else:
                    add_to_valid_data = False
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
                if add_to_valid_data:
                    valid_data.append(your_persona_description + " # " + question[0])
                    valid_data.append(question_line + " # " + answer[0])
                if add_to_test_data:
                    test_data.append(question[0])
                    test_data.append(answer[0])
                if context_pair_counter < context_pair_count or context_pair_count == 0:
                    question_line += " # " + question[0]
                    context_pair_counter += 1
                else:
                    question_line = your_persona_description + " # " + question[0]
                    context_pair_counter = 0
                answer_line = question_line + " # " + answer[0]
                train_data.append(question_line)
                train_data.append(answer_line)
                question_line = answer_line

    return train_data, valid_data, test_data


def tokenize(text: string, t):
    tokens = [tok for tok in t.tokenizer(text) if not tok.text.isspace()]
    text_tokens = [tok.text for tok in tokens]
    return tokens, text_tokens


def tokenize_and_join(text, t, jointoken=JOIN_TOKEN):
    tokenized_text = []
    for sentnence in text:
        tokenized_text.append(jointoken.join(tokenize(sentnence, t)[1]))
    return tokenized_text


def prepare_data():
    print("Prepare data")
    if DATA_TYPE == "PERSONA":
        train_data, valid_data, test_data = prepare_Persona_chat('persona_chat.txt', CONTEXT_PAIR_COUNT)
    elif DATA_TYPE == "TWITTER":
        train_data, valid_data, test_data = prepare_Twitter_data('twitter_chat.txt')

    tokenized_train_data = tokenize_and_join(train_data, nlp)
    tokenized_valid_data = tokenize_and_join(valid_data, nlp)
    tokenized_test_data = tokenize_and_join(test_data, nlp)

    print("train data: ", len(tokenized_train_data))
    print("valid data: ", len(tokenized_valid_data))
    print("test data: ", len(tokenized_test_data))

    save_to_csv('train.csv', tokenized_train_data)
    save_to_csv('valid.csv', tokenized_valid_data)
    save_to_csv('test.csv', tokenized_test_data)


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


def greedy_decode(model, vocab, fields, trg_indexes, hidden, cell, max_len):
    trg = []
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        predicted, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = predicted.argmax(1).item()
        trg.append(pred_token)
        if pred_token == vocab.stoi[fields['answer'].eos_token]:
            break
    return [vocab.itos[i] for i in trg]


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
    beam_width = 3
    topk = 2  # how many sentence do you want to generate
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
            if qsize > 100: break

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
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim,
                            self.n_layers)  # hidden state produced in the previous time step and the word embedding of the final output from the previous time step
        self.classifier = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def forward(self, input_seq, hidden, encoder_outputs):
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
        embeds_q = self.embedder(input_sequence).to(device)
        enc_q = self.dropout(embeds_q).to(device)
        outputs, (hidden, cell) = self.lstm(enc_q)
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
    def __init__(self, config, vocab):
        super().__init__()
        self.encoder = Encoder(config, vocab).to(device)
        if WITH_ATTENTION:
            self.decoder = LuongDecoder(config, vocab).to(device)
        else:
            self.decoder = Decoder(config, vocab).to(device)

        assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src [seq_len, batch_size]
        # trg [seq_len, batch_size]
        max_len, batch_size = trg.size()
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(device)
        enc_output, hidden, cell = self.encoder(src)
        enc_output = enc_output.permute(1, 0, 2)
        decoder_h = (hidden, cell)
        decoder_input = trg[0, :]

        for t in range(1, max_len):
            if WITH_ATTENTION:
                output, decoder_h, attn_weights = self.decoder(decoder_input, decoder_h, enc_output)
            else:
                output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[t] = output
            use_teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            decoder_input = (trg[t] if use_teacher_force else top1)

        return outputs.to(device)


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
        clip_grad_norm_(model.parameters(), 0.5)
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
    output, hidden, cell = model.encoder(src_tensor)
    trg_indexes = [vocab.stoi[fields['answer'].init_token]]
    trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
    if IS_BEAM_SEARCH:
        trg_tensor = beam_decode(model.decoder, vocab, fields, trg_tensor, hidden, cell)
    else:
        trg_tensor = greedy_decode(model, vocab, fields, trg_tensor, hidden, cell, 10)
    return trg_tensor


def train_model(model, fields, train_iter, valid_iter):
    print("Train model")
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters())
    pad_idx = fields['question'].vocab.stoi[fields['answer'].pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    best_validation_loss = float('inf')
    experiment_name = "train_" + time.strftime('%d-%m-%Y_%H:%M:%S')
    tensorboard_log_dir = './tensorboard-logs/{}/'.format(experiment_name)
    writer = SummaryWriter(tensorboard_log_dir)
    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_iter, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iter, criterion)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('valid_loss', valid_loss, epoch)
        for name, param in model.named_parameters():
            if param.grad is not None and not param.grad.data.is_sparse:
                writer.add_histogram(f"gradients_wrt_hidden_{name}/",
                                     param.grad.data.norm(p=2, dim=0),
                                     global_step=epoch)
        if valid_loss < best_validation_loss:
            best_validation_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(
                f'| Epoch: {epoch + 1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')


def main():
    if DEBUG:
        prepare_data()
        exit()

    config = {"train_batch_size": 5, "optimize_embeddings": False,
              "embedding_dim": 100, "hidden_dim": 256, "dropout_rate": 0.1, "num_layers": 4,
              "attention_model": 'concat'}

    # Specify Fields in our dataset
    data_fields = [('question', TEXT), ('answer', TEXT)]
    fields = dict(data_fields)

    # Build the dataset for train, validation and test sets
    trn, vld, test = TabularDataset.splits(
        path=".",  # the root directory where the data lies
        train='train.csv', validation="valid.csv", test='test.csv',
        format='csv',
        skip_header=True,
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
    for i, batch in enumerate(train_iter):
        if i < 2:
            print(batch)
        else:
            break

    model = Seq2Seq(config, vocab)

    if IS_TEST:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device(device)))
        test_data = load_csv('test.csv')
        data_to_save = []
        for i in range(0, len(test_data), 2):
            answer = test_model(test_data[i], fields, vocab, model)
            data_to_save.append(test_data[i])
            data_to_save.append(answer)
            print("QUESTION: ", test_data[i])
            print("ANSWER: ", answer)
        filename_timestamp = time.strftime('%d-%m-%Y_%H:%M:%S') + ".csv"
        save_to_csv(filename_timestamp, data_to_save)
    else:
        train_model(model, fields, train_iter, valid_iter)


if __name__ == "__main__":
    main()
