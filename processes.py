import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torchtext.data import BucketIterator
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.vocab import GloVe
from tqdm import tqdm

from model_scripts.decoding_algorithms import greedy_decode, beam_decode, beam_decode_mixed
from model_scripts.lm import LM
from model_scripts.seq2seq import Seq2Seq
from preprocessing import *
from utils.model_save_load import save_best_model

TEXT = Field(sequential=True, tokenize=lambda s: str.split(s, sep=JOIN_TOKEN), include_lengths=True,
             init_token='<sos>', eos_token='<eos>', pad_token='<pad>', lower=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal(param.data, mean=0, std=0.01)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


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
            output = model(src, trg, TEXT.vocab.stoi[TEXT.pad_token])
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
                output = model(src, trg, TEXT.vocab.stoi[TEXT.pad_token])
            else:
                trg = batch.source
                output = model(trg)

            # first output is 00s
            # the last iteration is not done, therefore we do not need to throw away the last output

            scores = output[1:].view(-1, output.shape[2])
            targets = trg[0][1:].view(-1)

            pad_mask = targets != TEXT.vocab.stoi[TEXT.pad_token]
            # filter out pads
            scores = scores[pad_mask]
            targets = targets[pad_mask]

            # trg shape shape is [(sequence_len - 1) * batch_size]
            # output shape is [(sequence_len - 1) * batch_size, output_dim]
            loss = criterion(scores, targets)

            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def fit_model(model, train_iter, valid_iter, n_epochs, clip, model_path):
    """

    Args:
        model: the model for training
        train_iter: train iterator
        valid_iter: valid iterator
        n_epochs: epochs number
        clip: value for clipping
        model_path: path to save best model

    Returns: None

    """
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
            save_best_model(model, model_path)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    model.tb.close()


def test_model(nlp, example, vocab, config, models, stylized_score_tensors, weights):
    """

    Args:
        nlp: tokenizer
        example: example for testing a trained model
        vocab: vocabulary
        config: configuration
        models: stylized models
        stylized_score_tensors: stylized tensors for feature-based decoding
        weights: stylized weights

    Returns: the generated seuqence

    """
    _, tokenized = tokenize(example, nlp)
    tokenized = [TEXT.init_token] + tokenized + [TEXT.eos_token]
    numericalized = [vocab.stoi[t] for t in tokenized]
    src_tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)
    enc_output, hidden, cell = models[0].encoder(src_tensor)
    enc_output = enc_output.permute(1, 0, 2)
    trg_indexes = [vocab.stoi[TEXT.init_token]]
    trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
    sos_token = vocab.stoi[TEXT.init_token]
    eos_token = vocab.stoi[TEXT.eos_token]

    if config["decoding_type"] == "beam":
        trg_tensor = beam_decode(vocab, config['beam_width'], config['max_sentence_len'], config['max_sentences'],
                                 models[0].decoder, config["with_attention"], trg_tensor,
                                 (hidden, cell), enc_output, sos_token, eos_token, device)

    elif config["decoding_type"] == "weighted_beam":
        trg_tensor = beam_decode_mixed(vocab, config['beam_width'], config['max_sentence_len'], config['max_sentences'],
                                       models, config["with_attention"], weights, stylized_score_tensors,
                                       trg_tensor, (hidden, cell), enc_output, sos_token, eos_token, device)

    else:
        trg_tensor = greedy_decode(vocab, models[0].decoder, config["with_attention"], trg_tensor, hidden, cell,
                                   enc_output, eos_token, device, config['max_sentence_len'])
    return trg_tensor


def run_train_lm_model(config):
    # Specify Fields in dataset
    data_fields = [('source', TEXT), ('target', TEXT)]
    fields = dict(data_fields)

    # Build the dataset for train, validation and test sets
    trn, vld = TabularDataset.splits(
        path="./.data",  # the root directory where the data lies
        train='pre_train.csv', validation="pre_valid.csv",
        format='csv',
        skip_header=True,
        fields=data_fields)

    # Build vocabulary
    print("[Building vocabulary]")
    fields["source"].build_vocab(vld, vectors=GloVe(name='6B', dim=config["embedding_dim"]))
    vocab = fields["source"].vocab
    print("[Length of vocabulary: ", len(vocab), "]")

    print("[Train ", config["style"], " Language model]")
    model = LM(config, vocab, device)  # .to(device)
    data_fields = [('source', TEXT)]
    if config["style"] == 'funny':
        train_filename = 'jokes_train.csv'
        valid_filename = 'jokes_valid.csv'
        save_path = MODEL_SAVE_FUNNY_PATH
    elif config["style"] == 'poetic':
        train_filename = 'poetic_train.csv'
        valid_filename = 'poetic_valid.csv'
        save_path = MODEL_SAVE_POETIC_PATH
    elif config["style"] == 'positive':
        train_filename = 'positive_train.csv'
        valid_filename = 'positive_valid.csv'
        save_path = MODEL_SAVE_POSITIVE_PATH
    elif config["style"] == 'negative':
        train_filename = 'negative_train.csv'
        valid_filename = 'negative_valid.csv'
        save_path = MODEL_SAVE_NEGATIVE_PATH

    # Build the dataset for train, validation and test sets
    trn, vld = TabularDataset.splits(
        path="./.data",  # the root directory where the data lies
        train=train_filename, validation=valid_filename,
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


def run_model(config):
    # Specify Fields in dataset
    data_fields = [('source', TEXT), ('target', TEXT)]
    fields = dict(data_fields)

    # Build the dataset for train, validation and test sets
    trn, vld = TabularDataset.splits(
        path=SAVE_DATA_PATH,  # the root directory where the data lies
        train='pre_train.csv', validation="pre_valid.csv",
        format='csv',
        skip_header=True,
        fields=data_fields)

    # Build vocabulary
    print("[Building vocabulary]")
    fields["source"].build_vocab(vld, vectors=GloVe(name='6B', dim=config["embedding_dim"]))
    vocab = fields["source"].vocab
    print("[Length of vocabulary: ", len(vocab), "]")

    if config["pretraining"] and config["process"] == 'train':
        print("[Pretraining]")
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
        model = Seq2Seq(config, vocab, device)
        fit_model(model, train_iter, valid_iter, config["n_epochs"], config["clip"], MODEL_PREPROCESS_SAVE_PATH)

    train, valid = TabularDataset.splits(
        path=SAVE_DATA_PATH,
        train='persona_train.csv', validation="persona_valid.csv",
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
                                shuffle=False, sort=False,
                                batch_size=config["train_batch_size"],
                                repeat=False,
                                device=device)

    if config["process"] == 'train':
        print("[Train model]")
        model = Seq2Seq(config, vocab, device)
        if config["with_pretrained_model"]:
            print("[Training with preprocess]")
            model.load_state_dict(torch.load(MODEL_PREPROCESS_SAVE_PATH, map_location=torch.device(device)))
        fit_model(model, train_iter, valid_iter, config["n_epochs"], config["clip"], MODEL_SAVE_PATH)

    elif config["process"] == 'test':
        style_scores_tensors = []
        models = []

        model = Seq2Seq(config, vocab, device)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device(device)))
        model.eval()
        models.append(model)
        weights = []
        if config['is_stylized_generation']:
            if config["with_stylized_lm"]:
                weights.append(float(config["baseline_weight"]))
                weights.append(float(config["jokes_weight"]))
                weights.append(float(config["poetic_weight"]))
                weights.append(float(config["positive_weight"]))
                weights.append(float(config["negative_weight"]))

                model_funny = LM(config, vocab, device)
                model_funny.load_state_dict(torch.load(MODEL_SAVE_FUNNY_PATH, map_location=torch.device(device)))
                model_funny.eval()

                model_poetic = LM(config, vocab, device)
                model_poetic.load_state_dict(torch.load(MODEL_SAVE_POETIC_PATH, map_location=torch.device(device)))
                model_poetic.eval()

                model_positive = LM(config, vocab, device)
                model_positive.load_state_dict(torch.load(MODEL_SAVE_POSITIVE_PATH, map_location=torch.device(device)))
                model_positive.eval()

                model_negative = LM(config, vocab, device)
                model_negative.load_state_dict(torch.load(MODEL_SAVE_NEGATIVE_PATH, map_location=torch.device(device)))
                model_negative.eval()

                models.append(model_funny)
                models.append(model_poetic)
                models.append(model_positive)
                models.append(model_negative)

            if config['with_controlling_attributes']:
                joke_scores_tensor = []
                jokes_dict = load_json(DATASETS_PATH + 'jokes_dict.json')
                for i in range(len(vocab)):
                    word = vocab.itos[i]
                    if word not in jokes_dict:
                        joke_scores_tensor.append(0)
                    else:
                        joke_scores_tensor.append(float(jokes_dict[word]))
                style_scores_tensors.append(torch.FloatTensor(joke_scores_tensor))

        test_data = load_csv(SAVE_DATA_PATH + 'persona_test.csv')
        data_to_save = []
        nlp = en_core_web_sm.load()
        nlp.tokenizer = create_custom_tokenizer(nlp)
        file_path = "./tests/" + time.strftime('%d-%m-%Y_%H:%M:%S') + ".csv"

        with open(file_path, mode='w') as csv_file:
            fieldnames = ['source', 'target']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(0, len(test_data), 2):
                answer = test_model(nlp, test_data[i], vocab, config, models, style_scores_tensors, weights)
                writer.writerow({'source': test_data[i], 'target': answer})
                data_to_save.append(test_data[i])
                data_to_save.append(answer)
                if i % 1000 == 0:
                    print("[SOURCE: ", test_data[i], "]")
                    print("[TARGET: ", answer, "]")
        csv_file.close()
        file_path = "./tests/" + time.strftime('%d-%m-%Y_%H:%M:%S') + ".csv"
        save_to_csv(file_path, data_to_save)
