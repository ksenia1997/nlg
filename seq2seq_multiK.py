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
    print("Train")
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
    print("Evaluate")
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

vocab = SRC.vocab
print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
model = Seq2Seq(config, vocab)
model_fit(model, train_iterator, valid_iterator)
