import random
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model_scripts.decoder import Decoder


class LM(nn.Module):
    def __init__(self, config, vocab, device):
        super().__init__()
        self.device = device
        self.decoder = Decoder(config, vocab, device)  # .to(device)
        experiment_name = "train_LM_" + config["style"] + "_model_" + time.strftime('%d-%m-%Y_%H:%M:%S')
        tensorboard_log_dir = './tensorboard-logs/{}/'.format(experiment_name)
        self.tb = SummaryWriter(tensorboard_log_dir)

    def forward(self, trg, hidden=None, cell=None, teacher_forcing_ratio=0.1):
        max_len, batch_size = trg[0].size()
        decoder_input = trg[0][0, :]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[t] = output
            use_teacher_force = random.random() > teacher_forcing_ratio
            top1 = output.argmax(dim=1)
            decoder_input = trg[0][t] if use_teacher_force else top1
        return outputs.to(self.device)
