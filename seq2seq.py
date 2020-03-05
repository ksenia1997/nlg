import math
import operator
import time
from queue import PriorityQueue

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import BucketIterator
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.vocab import GloVe
from tqdm import tqdm

from preprocessing import *

TEXT = Field(sequential=True, tokenize=lambda s: str.split(s, sep=JOIN_TOKEN), include_lengths=True,
             init_token='<sos>', eos_token='<eos>', pad_token='<pad>', lower=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def greedy_decode(decoder, with_attention, vocab, trg_indexes, hidden, cell, enc_output, max_len=100):
    trg = []
    enc_output = enc_output.permute(1, 0, 2)
    decoder_h = (hidden, cell)
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        if with_attention:
            predicted, decoder_h, attn_weights = decoder(trg_tensor, decoder_h, enc_output)
        else:
            predicted, hidden, cell = decoder(trg_tensor, hidden, cell)
        pred_token = predicted.argmax(1).item()
        trg.append(pred_token)
        trg_indexes = torch.cat((trg_indexes, torch.LongTensor([pred_token]).to(device)), 0)
        if pred_token == vocab.stoi[TEXT.eos_token]:
            break
    return [vocab.itos[i] for i in trg]


class BeamSearchNode(object):
    def __init__(self, hidden_state, previous_node, word_ids, log_prob, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hidden_state
        self.prev_node = previous_node
        self.word_ids = word_ids
        self.logp = log_prob
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp / float(self.length - 1 + 1e-6) + alpha * reward


def beam_decode_mixed(decoders, scores, vocab, beam_width, max_len, max_sentence_count, target_tensor, encoder_hiddens,
                      encoder_outputs):
    SOS_token = TEXT.init_token
    EOS_token = TEXT.eos_token
    # can be 1 or several decoders
    if len(decoders) != 1:
        assert len(decoders) == len(scores), "Number of decoders should equal to the number of scores"
    assert sum(scores) == 1, "Sum of scores must be 1"

    decoded_batch = []
    max_len = max(max_len, 2)
    beam_width = max(beam_width, 2)
    sentences = ""
    print("beam target tensor: ", target_tensor.size())
    for idx in range(target_tensor.size(0)):
        if isinstance(encoder_hiddens, tuple):
            decoder_hidden = (encoder_hiddens[0][:, idx, :].unsqueeze(1), encoder_hiddens[1][:, idx, :].unsqueeze(1))
        else:
            decoder_hidden = encoder_hiddens[:, idx, :].unsqueeze(1)
        encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)

        decoder_input = torch.LongTensor([vocab.stoi[SOS_token]]).to(device)
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes_queue = PriorityQueue()
        nodes_queue.put((-node.eval(), node))
        end_nodes = []
        while not nodes_queue.empty():
            score, n = nodes_queue.get()
            if len(end_nodes) >= max_sentence_count:
                break
            if n.length >= max_len or n.word_ids[-1] == vocab.stoi[EOS_token]:
                end_nodes.append((score, n))
                continue
            decoder_input = torch.LongTensor([n.word_ids[-1]]).to(device)
            decoder_hidden = n.h
            decoder_hs = []
            for i in range(len(decoders)):
                decoder_output, decoder_h, cell = decoders[i](decoder_input, decoder_hidden[0], decoder_hidden[1])
                decoder_hs.append((decoder_h, cell))
                if i == 0:
                    decoder_outputs = decoder_output
                else:
                    decoder_outputs = torch.cat((decoder_outputs, decoder_output), 0)
            decoder_outputs = decoder_outputs.transpose(0, 1)
            score = torch.FloatTensor(scores).to(device).unsqueeze(1)
            decoder_out_scored = torch.matmul(decoder_outputs, score).transpose(0, 1)

            for i in range(len(decoder_hs)):
                hidden = decoder_hs[i][0]
                cell = decoder_hs[i][1]
                if i == 0:
                    new_hidden = hidden * scores[i]
                    new_cell = cell * scores[i]
                else:
                    new_hidden = torch.add(new_hidden, hidden * scores[i])
                    new_cell = torch.add(new_cell, cell * scores[i])

            log_prob, indexes = torch.topk(decoder_out_scored, beam_width)
            print("log probs: ", log_prob.size(), indexes.size())
            decoder_hidden = (new_hidden, new_cell)
            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].unsqueeze(0)
                log_p = log_prob[0][new_k].item()
                decoded_t = torch.cat((n.word_ids, decoded_t), 0)
                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.length + 1)
                score = -node.eval()
                nodes_queue.put((score, node))

        for score, n in sorted(end_nodes, key=operator.itemgetter(0)):
            utterance = n.word_ids[::-1]
            sentence = [vocab.itos[i] for i in utterance]
            sentences += sentence + ["<eos>"]
            decoded_batch.append(sentence)
    return sentences


def beam_decode(decoder, vocab, target_tensor, decoder_hiddens, encoder_output, with_attention):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''
    print("Beam decode")
    beam_width = 3
    max_len = 5
    topk = 2  # how many sentence do you want to generate
    decoded_batch = []
    sentences = ""
    EOS_token = TEXT.eos_token
    SOS_token = TEXT.init_token

    # decoding goes sentence by sentence
    for idx in range(target_tensor.size(0)):
        decoder_hidden = decoder_hiddens

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([vocab.stoi[SOS_token]]).to(device)
        # Number of sentence to generate
        endnodes = []
        number_required = max(topk, 1)

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.word_ids
            decoder_hidden = n.h

            if (n.word_ids.item() == EOS_token and n.prev_node != None) or n.length >= max_len:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            if with_attention:
                predicted, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output)
            else:
                # decode for one step using decoder
                decoder_output, decoder_h, cell = decoder(decoder_input, decoder_hidden[0], decoder_hidden[1])
                decoder_hidden = (decoder_h, cell)
            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].unsqueeze(0)
                log_p = log_prob[0][new_k].item()
                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.length + 1)
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
            utterance.append(n.word_ids)
            # back trace
            while n.prev_node != None:
                n = n.prev_node
                utterance.append(n.word_ids)
            utterance = utterance[::-1]
            sentence = [vocab.itos[i] for i in utterance]
            sentences += ' '.join(sentence) + " <EOS>\n"
            decoded_batch.append(sentence)

    return sentences


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
                                                          padding_value=TEXT.vocab.stoi[TEXT.pad_token],
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

    def forward(self, inputs, hidden=None, cell=None):
        inputs = inputs.unsqueeze(0)
        # inputs [1,batch_size]
        embedded = self.embedder(inputs).to(device)
        embedded = self.dropout(embedded).to(device)
        # embedded [1, batch_size, embedded_dim]
        if hidden == None or cell == None:
            output, (hidden, cell) = self.lstm(embedded)
        else:
            output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predicted = F.log_softmax(self.linear(output.squeeze(0)).to(device), dim=1)
        return predicted, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.encoder = Encoder(config, vocab).to(device)
        self.with_attention = config["with_attention"]
        self.teacher_forcing_ratio = config["teacher_forcing_ratio"]
        if self.with_attention:
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

    def forward(self, src, trg):
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
            # print("IN: " + " ".join([TEXT.vocab.itos[x] for x in decoder_input.tolist()]))
            if self.with_attention:
                output, decoder_h, attn_weights = self.decoder(decoder_input, decoder_h, enc_output)
            else:
                output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[t] = output
            use_teacher_force = random.random() > self.teacher_forcing_ratio
            top1 = output.argmax(dim=1)
            decoder_input = trg[0][t] if use_teacher_force else top1
            # print("NEXT: " + " ".join([TEXT.vocab.itos[x] for x in decoder_input.tolist()]))
        return outputs.to(device)


class LM(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.decoder = Decoder(config, vocab).to(device)
        self.num_layers = config["num_layers"]
        self.batch_size = config["train_batch_size"]
        self.hidden_dim = config["hidden_dim"]
        experiment_name = "train_" + config["style"] + "_model_" + time.strftime('%d-%m-%Y_%H:%M:%S')
        tensorboard_log_dir = './tensorboard-logs/{}/'.format(experiment_name)
        self.tb = SummaryWriter(tensorboard_log_dir)

    def forward(self, trg, hidden=None, cell=None, teacher_forcing_ratio=0.1):
        max_len, batch_size = trg[0].size()
        decoder_input = trg[0][0, :]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(device)
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[t] = output
            use_teacher_force = random.random() > teacher_forcing_ratio
            top1 = output.argmax(dim=1)
            decoder_input = trg[0][t] if use_teacher_force else top1
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
    print("Train")
    #  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
    model.train()
    # loss
    epoch_loss = 0
    for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
        optimizer.zero_grad()
        # trg is of shape [sequence_len, batch_size]
        # output is of shape [sequence_len, batch_size, output_dim]
        if model.__class__.__name__ == "Seq2Seq":
            src = batch.source
            trg = batch.target
            output = model(src, trg)
        else:
            trg = batch.source
            output = model(trg)

        # first output are 00s
        # the last iteration is not done, therefore we do not need to throw away the last output

        scores = output[1:].view(-1, output.shape[2])
        targets = trg[0][1:].view(-1)

        pad_mask = targets != TEXT.vocab.stoi[TEXT.pad_token]
        # filter out pads
        scores = scores[pad_mask]
        targets = targets[pad_mask]

        # trg shape shape should be [(sequence_len - 1) * batch_size]
        # output shape should be [(sequence_len - 1) * batch_size, output_dim]
        loss = criterion(scores, targets)
        # backward pass
        loss.backward()

        # clip the gradients
        clip_grad_norm_(model.parameters(), clip)

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
            # trg shape shape should be [(sequence_len - 1) * batch_size]
            # output shape should be [(sequence_len - 1) * batch_size, output_dim]
            if model.__class__.__name__ == "Seq2Seq":
                src = batch.source
                trg = batch.target
                output = model(src, trg)
            else:
                trg = batch.source
                output = model(trg)

            # first output are 00s
            # the last iteration is not done, therefore we do not need to throw away the last output

            scores = output[1:].view(-1, output.shape[2])
            targets = trg[0][1:].view(-1)

            pad_mask = targets != TEXT.vocab.stoi[TEXT.pad_token]
            # filter out pads
            scores = scores[pad_mask]
            targets = targets[pad_mask]

            # trg shape shape should be [(sequence_len - 1) * batch_size]
            # output shape should be [(sequence_len - 1) * batch_size, output_dim]
            loss = criterion(scores, targets)

            epoch_loss += loss.item()
            # if (i + 1) % 100 == 0:
            #    print("eval loss: ", epoch_loss / i)
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def fit_model(model, train_iter, valid_iter, n_epochs, clip, model_path):
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    best_validation_loss = float('inf')

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_iter, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        model.tb.add_scalar('train_loss', train_loss, epoch)
        model.tb.add_scalar('valid_loss', valid_loss, epoch)

        for name, param in model.named_parameters():
            if param.grad is not None and not param.grad.data.is_sparse:
                model.tb.add_histogram(f"gradients_wrt_hidden_{name}/",
                                       param.grad.data.norm(p=2, dim=0),
                                       global_step=epoch)
        if valid_loss < best_validation_loss:
            best_validation_loss = valid_loss
            torch.save(model.state_dict(), model_path)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    model.tb.close()


def test_model(example, vocab, model, config):
    model.eval()
    nlp = en_core_web_sm.load()
    nlp.tokenizer = create_custom_tokenizer(nlp)
    _, tokenized = tokenize(example, nlp)
    tokenized = [TEXT.init_token] + tokenized + [TEXT.eos_token]
    numericalized = [vocab.stoi[t] for t in tokenized]
    src_tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)
    enc_output, hidden, cell = model.encoder(src_tensor)
    trg_indexes = [vocab.stoi[TEXT.init_token]]
    trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
    if config["decoding_type"] == "beam":
        trg_tensor = beam_decode(model.decoder, vocab, trg_tensor, (hidden, cell), enc_output, config["with_attention"])
    elif config["decoding_type"] == "weighted_beam":
        trg_tensor = beam_decode_mixed([model.decoder], [0.4, 0.6], vocab, 3, 10, 2, trg_tensor, (hidden, cell),
                                       enc_output)
    else:
        trg_tensor = greedy_decode(model.decoder, config["with_attention"], vocab, trg_tensor, hidden, cell, enc_output,
                                   100)
    return trg_tensor


def run_seq2seq(config):
    if config["prepare_data"] or config["data_BART"]:
        prepare_data(config)
        exit()

    if config["prepare_dict"]:
        prepare_dict(config)
        exit()

    # Specify Fields in dataset
    data_fields = [('source', TEXT), ('target', TEXT)]
    fields = dict(data_fields)

    # Build the dataset for train, validation and test sets
    trn, vld, test = TabularDataset.splits(
        path="./.data",  # the root directory where the data lies
        train='train.csv', validation="valid.csv", test='test.csv',
        format='csv',
        skip_header=True,
        fields=data_fields)

    # Build vocabulary
    print("Building vocabulary")
    fields["source"].build_vocab(trn, vectors=GloVe(name='6B', dim=config["embedding_dim"]))
    vocab = fields["source"].vocab
    print("len vocab: ", len(vocab))
    if config["train_preprocess"]:
        train, valid = TabularDataset.splits(
            path="./.data",
            train='twitter_train.csv', validation="twitter_valid.csv",
            format='csv',
            skip_header=True,
            fields=data_fields)
        # Create a set of iterators
        train_iter = BucketIterator(train,
                                    shuffle=True, sort=False,
                                    batch_size=config["train_batch_size"],
                                    repeat=False,
                                    device=device)
        valid_iter = BucketIterator(valid,
                                    shuffle=True, sort=False,
                                    batch_size=config["train_batch_size"],
                                    repeat=False,
                                    device=device)
        model = Seq2Seq(config, vocab)
        fit_model(model, train_iter, valid_iter, config["n_epochs"], config["clip"], MODEL_PREPROCESS_SAVE_PATH)

    if config["process"] != 'train_lm':
        # Create a set of iterators
        train_iter = BucketIterator(trn,
                                    shuffle=True, sort=False,
                                    batch_size=config["train_batch_size"],
                                    repeat=False,
                                    device=device)
        valid_iter = BucketIterator(vld,
                                    shuffle=False, sort=False,
                                    batch_size=config["train_batch_size"],
                                    repeat=False,
                                    device=device)

        # print("Most common: ", vocab.freqs.most_common(50))
        for i, batch in enumerate(train_iter):
            if i < 2:
                print(batch)
            else:
                break

    if config["process"] == 'test':
        model = Seq2Seq(config, vocab)
        jokes_dict = load_json(DATA_PATH + 'jokes_dict.json')
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device(device)))
        test_data = load_csv('datasets/test.csv')
        data_to_save = []

        for i in range(0, len(test_data), 2):
            answer = test_model(test_data[i], vocab, model, config)
            answer_str = ""
            for a in answer:
                answer_str += a + " "
            data_to_save.append(test_data[i])
            data_to_save.append(answer_str)
            if i % 1000 == 0:
                print("SOURCE: ", test_data[i])
                print("TARGET: ", answer_str)
        file_path = "./tests/" + time.strftime('%d-%m-%Y_%H:%M:%S') + ".csv"
        save_to_csv(file_path, data_to_save)

    elif config["process"] == 'train':
        if config["with_preprocess"]:
            model.load_state_dict(torch.load(MODEL_PREPROCESS_SAVE_PATH, map_location=torch.device(device)))
        else:
            model = Seq2Seq(config, vocab)
        fit_model(model, train_iter, valid_iter, config["n_epochs"], config["clip"], MODEL_SAVE_PATH)
    elif config["process"] == 'train_lm':
        print("Train Language model")
        model = LM(config, vocab).to(device)
        data_fields = [('source', TEXT)]
        if config["style"] == 'funny':
            train_filename = 'jokes_train.csv'
            valid_filename = 'jokes_valid.csv'
            test_filename = 'jokes_test.csv'
            save_path = MODEL_SAVE_FUNNY_PATH

        # Build the dataset for train, validation and test sets
        trn, vld, test = TabularDataset.splits(
            path="./.data",  # the root directory where the data lies
            train=train_filename, validation=valid_filename, test=test_filename,
            format='csv',
            skip_header=True,
            fields=data_fields)
        # Create a set of iterators
        train_iter = BucketIterator(trn,
                                    shuffle=True, sort=False,
                                    batch_size=config["train_batch_size"],
                                    repeat=False,
                                    device=device)
        valid_iter = BucketIterator(vld,
                                    shuffle=False, sort=False,
                                    batch_size=config["train_batch_size"],
                                    repeat=False,
                                    device=device)
        for i, batch in enumerate(train_iter):
            if i < 2:
                print(batch)
            else:
                break
        fit_model(model, train_iter, valid_iter, config["n_epochs"], config["clip"], save_path)
