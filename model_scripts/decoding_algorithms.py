import operator
from queue import PriorityQueue

import numpy as np
import torch

from params import JOIN_TOKEN


def greedy_decode(vocab, decoder, with_attention, trg_indexes, hidden, cell, enc_output, eos_token, device,
                  max_len=100):
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
        if pred_token == eos_token:
            break
    return JOIN_TOKEN.join([vocab.itos[i] for i in trg])


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
        reward = np.random.uniform(0.1, 10 ** (-20))
        # Add here a function for shaping a reward
        return self.logp / float(self.length - 1 + 1e-6) + alpha * reward


def beam_decode_mixed(vocab, beam_width, max_len, max_sentence_count, models, with_attention, decoder_scores,
                      stylized_score_tensors, target_tensor, encoder_hiddens, encoder_outputs, sos_token, eos_token,
                      device):
    # can be 1 or several decoders
    if len(models) != 1:
        assert len(models) == len(decoder_scores), "Number of model decoders should be equal to the number of scores"
    else:
        assert len(decoder_scores) - 1 == len(
            stylized_score_tensors), "Number of styles should be equal to number of discrete distribution "

    assert sum(decoder_scores) == 1, "Sum of scores must be 1"

    decoded_batch = []
    max_len = max(max_len, 2)
    beam_width = max(beam_width, 2)
    sentences = ""
    for idx in range(target_tensor.size(0)):
        if isinstance(encoder_hiddens, tuple):
            decoder_hidden = (encoder_hiddens[0][:, idx, :].unsqueeze(1), encoder_hiddens[1][:, idx, :].unsqueeze(1))
        else:
            decoder_hidden = encoder_hiddens[:, idx, :].unsqueeze(1)
        encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)

        decoder_input = torch.LongTensor([sos_token]).to(device)
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes_queue = PriorityQueue()
        nodes_queue.put((-node.eval(), node))
        end_nodes = []
        while not nodes_queue.empty():
            score, n = nodes_queue.get()
            if len(end_nodes) >= max_sentence_count:
                break
            if n.length >= max_len or n.word_ids[-1] == eos_token:
                end_nodes.append((score, n))
                continue
            decoder_input = torch.LongTensor([n.word_ids[-1]]).to(device)
            decoder_hidden = n.h
            decoder_hs = []
            for i in range(len(models)):
                if models[i].__class__.__name__ == "Seq2Seq" and with_attention:
                    decoder_output, (decoder_h, cell), _ = models[i].decoder(decoder_input, decoder_hidden,
                                                                             encoder_output)
                else:
                    decoder_output, decoder_h, cell = models[i].decoder(decoder_input, decoder_hidden[0],
                                                                        decoder_hidden[1])
                decoder_hs.append((decoder_h, cell))
                if i == 0:
                    decoder_outputs = decoder_output
                else:
                    decoder_outputs = torch.cat((decoder_outputs, decoder_output), 0)
            if len(models) == 1:
                dec_output = decoder_outputs
                for i in range(len(stylized_score_tensors)):
                    stylized_out = []
                    # stylized_score_tensors is a list of tensors of size [len(vocab)]
                    for coef in range(stylized_score_tensors[i].size(0)):
                        stylized_out.append(torch.exp(stylized_score_tensors[i][coef]) + dec_output[0][coef])
                    stylized_out = torch.FloatTensor(stylized_out).to(device).unsqueeze(0)
                    decoder_outputs = torch.cat((decoder_outputs, stylized_out), 0)
            decoder_outputs = decoder_outputs.transpose(0, 1)

            score = torch.FloatTensor(decoder_scores).to(device).unsqueeze(1)
            decoder_out_scored = torch.matmul(decoder_outputs, score).transpose(0, 1)
            for i in range(len(decoder_hs)):
                hidden = decoder_hs[i][0]
                cell = decoder_hs[i][1]
                if i == 0:
                    new_hidden = hidden * decoder_scores[i]
                    new_cell = cell * decoder_scores[i]
                else:
                    new_hidden = torch.add(new_hidden, hidden * decoder_scores[i])
                    new_cell = torch.add(new_cell, cell * decoder_scores[i])

            log_prob, indexes = torch.topk(decoder_out_scored, beam_width)
            decoder_hidden = (new_hidden, new_cell)
            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].unsqueeze(0)
                log_p = log_prob[0][new_k].item()
                decoded_t = torch.cat((n.word_ids, decoded_t), 0)
                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.length + 1)
                score = -node.eval()
                nodes_queue.put((score, node))

        for score, n in sorted(end_nodes, key=operator.itemgetter(0)):
            utterance = n.word_ids
            sentence = [vocab.itos[i] for i in utterance]
            sentences += JOIN_TOKEN.join(sentence) + "<eos>"
            decoded_batch.append(sentence)
    return sentences


def beam_decode(vocab, beam_width, max_len, topk, decoder, with_attention, target_tensor, decoder_hiddens,
                encoder_output, sos_token, eos_token, device):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''
    print("Beam decode")
    decoded_batch = []
    sentences = ""

    # decoding goes sentence by sentence
    for idx in range(target_tensor.size(0)):
        decoder_hidden = decoder_hiddens

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([sos_token]).to(device)
        # Number of sentence to generate
        endnodes = []
        number_required = max(topk, 1)

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))

        # start beam search
        while True:
            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.word_ids
            decoder_hidden = n.h

            if (n.word_ids.item() == eos_token and n.prev_node != None) or n.length >= max_len:
                endnodes.append((score, n))
                # if we reached maximum of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            if with_attention:
                decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output)
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

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = [n.word_ids]
            # back trace
            while n.prev_node is not None:
                n = n.prev_node
                utterance.append(n.word_ids)
            utterance = utterance[::-1]
            sentence = [vocab.itos[i] for i in utterance]
            sentences += ' '.join(sentence) + " <eos>\n"
            decoded_batch.append(sentence)

    return sentences
