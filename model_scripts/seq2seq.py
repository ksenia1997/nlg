import random
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model_scripts.decoder import Decoder, LuongDecoder
from model_scripts.encoder import Encoder


class Seq2Seq(nn.Module):
    def __init__(self, config, vocab, device):
        super().__init__()
        self.device = device
        self.encoder = Encoder(config, vocab, device).to(device)
        self.with_attention = config["with_attention"]
        self.teacher_forcing_ratio = config["teacher_forcing_ratio"]
        if self.with_attention:
            self.decoder = LuongDecoder(config, vocab).to(device)
        else:
            self.decoder = Decoder(config, vocab, device).to(device)

        assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

        experiment_name = "train_seq2seq_" + time.strftime('%d-%m-%Y_%H:%M:%S')
        tensorboard_log_dir = './tensorboard-logs/{}/'.format(experiment_name)
        self.tb = SummaryWriter(tensorboard_log_dir)

    def forward(self, src, trg, pad_token):
        # src [seq_len, batch_size]
        # trg [seq_len, batch_size]
        print("src seq2seq: ", src.size())
        print("trg seq2seq: ", trg.size())

        max_len, batch_size = trg[0].size()
        outputs = torch.zeros(max_len, batch_size, self.decoder.output_dim).to(self.device)
        enc_output, hidden, cell = self.encoder(src, pad_token)
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
            use_teacher_force = random.random() < self.teacher_forcing_ratio
            top1 = output.argmax(dim=1)
            decoder_input = trg[0][t] if use_teacher_force else top1
            # print("NEXT: " + " ".join([TEXT.vocab.itos[x] for x in decoder_input.tolist()]))
        return outputs.to(self.device)
