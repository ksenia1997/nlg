import copy
import json
import logging
import math
import operator
import os
import pickle
from queue import PriorityQueue
from typing import List

import gpt.src.encoder as gpt2_encoder
import gpt.src.model as gpt2_model
import numpy as np
import tensorflow as tf
import torch
from fairseq import search
from sequence_generator import EnsembleModel
from sequence_generator import SequenceGenerator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def sample(model, tf_idf, sentences: List[str], beam: int = 1, verbose: bool = False,
           **kwargs) -> str:
    input = [model.encode(sentence) for sentence in sentences]
    hypos = generate(model, tf_idf, input, beam, verbose, **kwargs)
    return [model.decode(x['tokens']) for x in hypos]


def create_tf_idf(model):
    with open("../.data/tf-idf", "rb") as fp:
        sentences = pickle.load(fp)
    indexes = list(range(0, len(model.task.target_dictionary)))
    sentences_length = len(sentences)
    indexes_dict = dict(zip(indexes, [1 for x in range(0, len(indexes))]))
    for sentence in sentences:
        enc_sentence = model.encode(sentence)
        unique_tokens = np.unique(np.array(enc_sentence))
        for token in unique_tokens:
            indexes_dict[token] += 1
    matrix_tf_idf = torch.arange(len(model.task.target_dictionary)).type(torch.FloatTensor)
    for k, v in indexes_dict.items():
        matrix_tf_idf[k] = sentences_length / v
    max_nidf = torch.max(matrix_tf_idf)
    min_nidf = torch.min(matrix_tf_idf)
    matrix_tf_idf = torch.log(matrix_tf_idf) - min_nidf
    matrix_tf_idf = matrix_tf_idf / (max_nidf - min_nidf)
    return matrix_tf_idf


def get_bart_tensor_with_gpt2_idxs():
    with open("bart_arr_gpt2_idxs", "rb") as fp:
        bart_tensor_with_gpt2_idxs = pickle.load(fp)
    return  bart_tensor_with_gpt2_idxs

def convert_gpt_idxs_to_bart(logp, bart_vocab_size):
    bart_tensor_with_gpt2_idxs = get_bart_tensor_with_gpt2_idxs()
    converted_logp = []
    for i in range(bart_vocab_size):
        gpt2_idx = None
        if i in bart_tensor_with_gpt2_idxs:
            gpt2_idx = bart_tensor_with_gpt2_idxs[i]
        if gpt2_idx == 50257 or gpt2_idx is None:
            converted_logp.append(0)
        else:
            converted_logp.append(logp[0][gpt2_idx])
    return torch.tensor(converted_logp, device=device).unsqueeze(0)


def sample_beam(model, tf_idf, sentences: List[str], beam: int = 1, max_len: int = 100, temperature=1.,
                unk_penalty=0.001):
    pass


def generate(model, tf_idf_matrix, tokens: List[torch.LongTensor], beam: int = 5, verbose: bool = False,
             **kwargs) -> torch.LongTensor:
    sample = model._build_sample(tokens)

    # build generator using current args as well as any kwargs
    gen_args = copy.copy(model.args)
    gen_args.beam = beam
    for k, v in kwargs.items():
        setattr(gen_args, k, v)

    sampling = getattr(gen_args, 'sampling', False)
    sampling_topk = getattr(gen_args, 'sampling_topk', -1)
    sampling_topp = getattr(gen_args, 'sampling_topp', -1.0)
    diverse_beam_groups = getattr(gen_args, 'diverse_beam_groups', -1)
    diverse_beam_strength = getattr(gen_args, 'diverse_beam_strength', 0.5),
    match_source_len = getattr(gen_args, 'match_source_len', False)
    diversity_rate = getattr(gen_args, 'diversity_rate', -1)

    if sampling:
        search_strategy = search.Sampling(model.task.target_dictionary, sampling_topk, sampling_topp)
    elif diverse_beam_groups > 0:
        search_strategy = search.DiverseBeamSearch(
            model.task.target_dictionary, diverse_beam_groups, diverse_beam_strength)
    elif match_source_len:
        # this is useful for tagging applications where the output
        # length should match the input length, so we hardcode the
        # length constraints for simplicity
        search_strategy = search.LengthConstrainedBeamSearch(
            model.task.target_dictionary, min_len_a=1, min_len_b=0, max_len_a=1, max_len_b=0,
        )
    elif diversity_rate > -1:
        search_strategy = search.DiverseSiblingsSearch(model.task.target_dictionary, diversity_rate)
    else:
        search_strategy = search.BeamSearch(model.task.target_dictionary)

    # generator = model.task.build_generator(gen_args)
    generator = SequenceGenerator(
        model.task.target_dictionary,
        beam_size=getattr(gen_args, 'beam', 5),
        max_len_a=getattr(gen_args, 'max_len_a', 0),
        max_len_b=getattr(gen_args, 'max_len_b', 200),
        min_len=getattr(gen_args, 'min_len', 1),
        normalize_scores=(not getattr(gen_args, 'unnormalized', False)),
        len_penalty=getattr(gen_args, 'lenpen', 1),
        unk_penalty=getattr(gen_args, 'unkpen', 0),
        tf_idf=tf_idf_matrix,
        temperature=getattr(gen_args, 'temperature', 1.),
        match_source_len=getattr(gen_args, 'match_source_len', False),
        no_repeat_ngram_size=getattr(gen_args, 'no_repeat_ngram_size', 0),
        search_strategy=search_strategy,
    )

    translations = model.task.inference_step(
        generator,
        [model.model],
        sample,
        prefix_tokens=sample['net_input']['src_tokens'].new_zeros((len(tokens), 1)).fill_(
            model.task.source_dictionary.bos()),
    )

    if verbose:
        src_str_with_unk = model.string(tokens)
        logger.info('S\t{}'.format(src_str_with_unk))

    # Process top predictions
    hypos = [x[0] for x in translations]
    hypos = [v for _, v in sorted(zip(sample['id'].tolist(), hypos))]
    return hypos


class BeamSearchNode(object):
    def __init__(self, previous_node, word_ids, log_prob, length):
        '''
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.prev_node = previous_node
        self.word_ids = word_ids
        self.logp = log_prob
        self.length = length

    def eval(self, alpha=1.0):
        reward = np.random.uniform(0.1, 10 ** (-20))
        # Add here a function for shaping a reward
        return self.logp / float(self.length - 1 + 1e-6) + alpha * reward


def bart_beam_decode(model, tf_idf, input_tokens, beam_width, max_len, max_sentence_count, temperature, unk_penalty):
    assert max_len > 2
    assert max_sentence_count > 1
    assert beam_width > 2

    decoded_batch = []
    sentences = ""

    # BART pad, unk and eos
    pad = model.task.target_dictionary.pad()
    unk = model.task.target_dictionary.unk()
    eos = model.task.target_dictionary.eos()
    ensemble_model = EnsembleModel([model.model])

    # GPT-2
    enc_gpt2 = gpt2_encoder.get_encoder('117M')
    hparams = gpt2_model.default_hparams()
    gpt2_eos = '<|endoftext|>'
    with open(os.path.join('gpt/src/models', '117M', 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    gpt2_len = hparams.n_ctx
    print("Input tokens", input_tokens)
    for i in range(len(input_tokens)):
        endnodes = []
        nodes = PriorityQueue()
        qsize = 1
        print("Input tokens i: ", input_tokens[i])
        # BART
        enc_tokens = [model.encode(input_tokens[i])]
        sample = model._build_sample(enc_tokens)
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }
        src_tokens = encoder_input['src_tokens']
        encoder_outs = ensemble_model.forward_encoder(encoder_input)
        target_tokens = src_tokens.new(1, 1).long().fill_(eos)
        node = BeamSearchNode(None, target_tokens, 0, 1)
        nodes.put((-node.eval(), node))

        while True:
            score, n = nodes.get()
            decoder_input = n.word_ids
            if (n.word_ids[0][-1].item() == eos and n.prev_node != None) or n.length >= max_len:
                endnodes.append((score, n))
                if len(endnodes) >= max_sentence_count:
                    break
                else:
                    continue

            lprobs, avg_attn_scores = ensemble_model.forward_decoder(
                decoder_input, encoder_outs, temperature=temperature)
            # lprobs = lprobs.add(tf_idf)
            lprobs[:, pad] = -math.inf  # never select pad
            lprobs[:, unk] -= unk_penalty  # apply unk penalty
            if n.prev_node is None:
                # GPT-2 start generation with eos
                start_token = [[enc_gpt2.encoder[gpt2_eos]]]
                print("START TOKEN: ", start_token)
                
            else:
                print("GPT2 dec: ", decoder_input.squeeze(0).size())
                start_token = []
                bart_gpt2_dict = get_bart_tensor_with_gpt2_idxs()
                #dec_input = model.decode(decoder_input.squeeze(0))
                for item in decoder_input.squeeze(0):
                    gpt2_item = bart_gpt2_dict[item]
                    if gpt2_item == 50257:
                        start_token.append(0)
                    else:
                        start_token.append(gpt2_item)
                start_token = [start_token]
                print("dec input for GPT2", start_token)
                #print("enc EOS: ", enc_gpt2.encoder[gpt2_eos])
                #print("enc_gpt2.encoder: ", enc_gpt2.encoder[gpt2_eos + ' hello how are you?'])
                #start_token = enc_gpt2.encoder[gpt2_eos + dec_input]
                #start_token = torch.tensor(dec_input, device=device)
                #print("START: ", start_token, start_token.size())
            context = tf.convert_to_tensor(start_token)
            print("context: ", context)
            lm_output = gpt2_model.model(hparams=hparams, X=context, past=None, reuse=tf.AUTO_REUSE)
            logits = lm_output['logits'][:, :, :hparams.n_vocab]
            # converting tf.Tensor to torch.tensor
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                logits = logits.eval()
            logits = torch.tensor(logits)
            logits = torch.squeeze(logits, 0)

            converted_logits = convert_gpt_idxs_to_bart(logits, lprobs.size(1))
            print("converted logits: ", converted_logits.size())
            lprobs = lprobs * 0.6
            add_probs = lprobs.add(converted_logits * 0.4)
            print("add probs: ", add_probs.size())
            log_prob, indexes = torch.topk(add_probs, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].unsqueeze(0).unsqueeze(0)
                print("decoded t: ", decoded_t, decoded_t.size())
                print("decoded input: ", decoder_input, decoder_input.size())
                decoded_t = torch.cat((decoder_input, decoded_t), 1)
                print("decoded t concatenated: ", decoded_t, decoded_t.size())
                log_p = log_prob[0][new_k].item()
                print("log p: ", log_p)
                node = BeamSearchNode(n, decoded_t, n.logp + log_p, n.length + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            for nn_i in range(len(nextnodes)):
                score, nn = nextnodes[nn_i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(max_sentence_count)]

        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = [n.word_ids]
            # back trace
            while n.prev_node is not None:
                n = n.prev_node
                utterance.append(n.word_ids)
                print("n word_ids: ", n.word_ids.size())
                print("squeeze: ", n.word_ids.squeeze(0).size())
                sentence = model.decode(n.word_ids.squeeze(0))
                print("decoded sentence: ", sentence)
                decoded_batch.append(sentence)
    return decoded_batch

