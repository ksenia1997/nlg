import math
import operator
import time
from queue import PriorityQueue

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import BucketIterator
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.vocab import GloVe

from preprocessing import *
from utils.create_histogram import *

# Create Field object
# TEXT = data.Field(tokenize = 'spacy', lower=True, include_lengths = True, init_token = '<sos>',  eos_token = '<eos>')
TEXT = Field(sequential=True, tokenize=lambda s: str.split(s, sep=JOIN_TOKEN), init_token='<sos>',
             eos_token='<eos>', pad_token='<pad>', lower=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


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
        nn.init.normal(param.data, mean=0, std=0.01)


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
        embeds_q = self.embedder(input_sequence[0]).to(device)
        enc_q = self.dropout(embeds_q).to(device)
        inp_packed = pack_padded_sequence(enc_q, input_sequence[1], batch_first=False, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(inp_packed)
        outputs, output_lengths = pad_packed_sequence(outputs, batch_first=False,
                                                      padding_value=input_sequence[0][0][0],
                                                      total_length=input_sequence[0].shape[0])
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


def train(model, iterator, optimizer, criterion):
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
    print("Train")
    #  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
    model.train()
    # loss
    epoch_loss = 0
    print("iterator: ", len(iterator))
    for i, batch in enumerate(iterator):
        src = batch.question
        trg = batch.answer

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


def evaluate(model, iterator, criterion):
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
    print("Evaluate")
    # we don't need to update the model parameters. only forward pass.
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.question
            trg = batch.answer

            output = model(src, trg, 0)  # turn off the teacher forcing

            # trg shape shape should be [(sequence_len - 1) * batch_size]
            # output shape should be [(sequence_len - 1) * batch_size, output_dim]
            loss = criterion(output[:-1].view(-1, output.shape[2]), trg[0][1:].view(-1))
            epoch_loss += loss.item()
            # if (i + 1) % 100 == 0:
            #    print("eval loss: ", epoch_loss / i)
    return epoch_loss / len(iterator)


def fit_model(model, fields, train_iter, valid_iter):
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters())
    pad_idx = fields['question'].vocab.stoi[fields['answer'].pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    best_validation_loss = float('inf')

    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_iter, optimizer, criterion)
        valid_loss = evaluate(model, valid_iter, criterion)
        model.tb.add_scalar('train_loss', train_loss, epoch)
        model.tb.add_scalar('valid_loss', valid_loss, epoch)
        for name, param in model.named_parameters():
            #print("name: ", name, param)
            if param.grad is not None and not param.grad.data.is_sparse:
                #print("param grad: ", param.grad)
                #print("data: ", param.grad.data)
                model.tb.add_histogram(f"gradients_wrt_hidden_{name}/",
                                       param.grad.data.norm(p=2, dim=0),
                                       global_step=epoch)
        if valid_loss < best_validation_loss:
            best_validation_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(
                f'| Epoch: {epoch + 1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')
    model.tb.close()


def test_model(example, fields, vocab, model):
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


def main():
    if PREPARE_DATA:
        prepare_data()
        exit()

    if CREATE_HISTOGRAM:
        columns = load_histogram_data('datasets/description.csv')
        yp_desc = columns['Your persona description length']
        pp_desc = columns['Partner\'s persona description length']
        utr1_length = columns['utterance1 length']
        utr2_length = columns['utterance2 length']
        yp_desc.sort()
        pp_desc.sort()
        utr1_length.sort()
        utr2_length.sort()
        plot_histogram('Histogram of your persona description lengths', 'number of words in description',
                       'number of descriptions',
                       yp_desc, 50, 'persona_desc.png')
        plot_histogram('Histogram of partner\'s persona description lengths', 'number of words in description',
                       'number of descriptions', pp_desc, 50, 'partner_desc.png')
        plot_histogram('Histogram of utterances\' lengths of the first person', 'number of words in utterance',
                       'number of utterances', utr1_length, 50, 'uttr1_length.png')
        plot_histogram('Histogram of utterances\' lengths of the first person', 'number of words in utterance',
                       'number of utterances', utr2_length, 50, 'uttr2_length.png')
        exit()

    # Specify Fields in our dataset
    data_fields = [('question', TEXT), ('answer', TEXT)]
    fields = dict(data_fields)

    if PREPROCESS:
        trn, vld = TabularDataset.splits(
            path="./datasets",  # the root directory where the data lies
            train='twitter_train.csv', validation="twitter_valid.csv",
            format='csv',
            skip_header=True,
            fields=data_fields)
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
        model = Seq2Seq(config, vocab)
        fit_model(model, fields, train_iter, valid_iter)

    # Build the dataset for train, validation and test sets
    trn, vld, test = TabularDataset.splits(
        path="./datasets",  # the root directory where the data lies
        train='train.csv', validation="valid.csv", test='test.csv',
        format='csv',
        skip_header=True,
        fields=data_fields)

    if not PREPROCESS:
        # Build vocabulary
        print("Build vocabulary")
        fields["question"].build_vocab(trn, vectors=GloVe(name='6B', dim=config["embedding_dim"]))
        vocab = fields["question"].vocab
        print("len vocab: ", len(vocab))

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

    print("train iter: ", len(train_iter))
    print("valid iter: ", len(valid_iter))
    # print("Most common: ", vocab.freqs.most_common(50))
    for i, batch in enumerate(train_iter):
        if i < 2:
            print(batch)
        else:
            break
    if PREPROCESS:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device(device)))
    else:
        model = Seq2Seq(config, vocab)

    if IS_TEST:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device(device)))
        test_data = load_csv('datasets/test.csv')
        data_to_save = []
        for i in range(0, len(test_data), 2):
            answer = test_model(test_data[i], fields, vocab, model)
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
    else:
        fit_model(model, fields, train_iter, valid_iter)


if __name__ == "__main__":
    main()
