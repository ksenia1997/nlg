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


def sample(model, idf, sentences: List[str], beam: int = 1, verbose: bool = False,
           **kwargs) -> str:
    input = [model.encode(sentence) for sentence in sentences]
    hypos = generate(model, idf, input, beam, verbose, **kwargs)
    return [model.decode(x['tokens']) for x in hypos]


def create_idf(model):
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
    return bart_tensor_with_gpt2_idxs


def convert_gpt_idxs_to_bart(logp, bart_vocab_size, bart_tensor_with_gpt2_idxs):
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


def generate(model, idf_matrix, tokens: List[torch.LongTensor], beam: int = 5, verbose: bool = False,
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
        idf=idf_matrix,
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
    def __init__(self, previous_node, word_ids, log_prob, length, penalty):
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
        self.block_penalty = penalty

    def eval(self, alpha=1.0):
        reward = np.random.uniform(0.1, 10 ** (-20))
        # Add here a function for shaping a reward
        return self.logp / float(self.length - 1 + 1e-6) + alpha * reward


class BartModel(object):
    def __init__(self, model):
        self.model = model
        self.pad = model.task.target_dictionary.pad()
        self.unk = model.task.target_dictionary.unk()
        self.eos = model.task.target_dictionary.eos()
        self.ensemble_model = EnsembleModel([model.model])


class GPT2Model(object):
    def __init__(self):
        self.encoder = gpt2_encoder.get_encoder('117M')
        self.hyper_params = gpt2_model.default_hparams()
        self.eos = '<|endoftext|>'
        self.bart_gpt2_dict = get_bart_tensor_with_gpt2_idxs()
        with open(os.path.join('gpt/src/models', '117M', 'hparams.json')) as f:
            self.hyper_params.override_from_dict(json.load(f))


def greedy_decoding(bart: BartModel, gpt2: GPT2Model, max_len=100):
    softmax = torch.nn.LogSoftmax()
    start_token = [gpt2.encoder.encoder[gpt2.eos]]
    decoded_items = torch.tensor((), dtype=torch.long)
    decoded_items.new(1, 1).long().fill_(bart.eos)
    print("dec items start: ", decoded_items)
    print("bart eos: ", bart.eos)
    for i in range(max_len):
        context = tf.convert_to_tensor([start_token])
        lm_output = gpt2_model.model(hparams=gpt2.hyper_params, X=context, past=None, reuse=tf.AUTO_REUSE)
        logits = lm_output['logits'][:, :, :gpt2.hyper_params.n_vocab]
        # converting tf.Tensor to torch.tensor
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            logits = logits.eval()
        logits = torch.tensor(logits)
        logits = torch.squeeze(logits, 0)

        converted_logits = convert_gpt_idxs_to_bart(logits, 50264, gpt2.bart_gpt2_dict)
        converted_logits = softmax(converted_logits)
        log_prob, indexes = torch.topk(converted_logits, 1)
        print("bart item: ", indexes[0][0].unsqueeze(0).unsqueeze(0))
        decoded_items = torch.cat((decoded_items.cuda(),  indexes[0][0].unsqueeze(0).unsqueeze(0).cuda()),1)
        gpt2_item = gpt2.bart_gpt2_dict[indexes[0][0]]
        print("gpt2 item: ", gpt2_item)
        start_token.append(gpt2_item)
        print("gpt2 start token ", start_token)
        print("bart dec items: ", decoded_items)
    decoded_gpt2 = gpt2.encoder.decode(start_token)
    decoded = bart.model.decode(decoded_items.squeeze(0))
    print("Decoded BART: ", decoded)
    print("Decoded GPT2: ", decoded_gpt2)




def bart_beam_decode(bart: BartModel, gpt2: GPT2Model, weights, input_tokens, beam_width=0, top_p=0.0, min_len=3,
                     max_len=100, max_sentence_count=2, temperature=1, unk_penalty=0.001, block_unigram=None):
    assert max_len > 2
    assert max_sentence_count > 1
    # assert beam_width > 2
    assert min_len > 1

    decoded_batch = []

    softmax = torch.nn.LogSoftmax()

    for i in range(len(input_tokens)):
        endnodes = []
        nodes = PriorityQueue()

        # BART
        enc_tokens = [bart.model.encode(input_tokens[i])]
        sample = bart.model._build_sample(enc_tokens)
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }
        src_tokens = encoder_input['src_tokens']
        encoder_outs = bart.ensemble_model.forward_encoder(encoder_input)
        target_tokens = src_tokens.new(1, 1).long().fill_(bart.eos)
        penalty = torch.zeros([1, 50264], dtype=torch.float64).to(device)
        node = BeamSearchNode(None, target_tokens, 0, 1, penalty)
        nodes.put((-node.eval(), node))

        while True:
            score, n = nodes.get()
            decoder_input = n.word_ids
            if (n.word_ids[0][-1].item() == bart.eos and n.prev_node != None) or n.length >= max_len:
                if n.length < min_len:
                    continue
                endnodes.append((score, n))
                if len(endnodes) >= max_sentence_count:
                    break
                else:
                    continue

            lprobs, avg_attn_scores = bart.ensemble_model.forward_decoder(
                decoder_input, encoder_outs, temperature=temperature)
            lprobs[:, bart.pad] = -math.inf  # never select pad
            lprobs[:, bart.unk] -= unk_penalty  # apply unk penalty

            if n.prev_node is None:
                start_token = [[gpt2.encoder.encoder[gpt2.eos]]]
            else:
                start_token = []
                for item in decoder_input.squeeze(0):
                    gpt2_item = gpt2.bart_gpt2_dict[item]
                    if gpt2_item == 50257:
                        start_token.append(0)
                    else:
                        start_token.append(gpt2_item)
                start_token = [start_token]

            context = tf.convert_to_tensor(start_token)
            lm_output = gpt2_model.model(hparams=gpt2.hyper_params, X=context, past=None, reuse=tf.AUTO_REUSE)
            logits = lm_output['logits'][:, :, :gpt2.hyper_params.n_vocab]
            # converting tf.Tensor to torch.tensor
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                logits = logits.eval()
            logits = torch.tensor(logits)
            logits = torch.squeeze(logits, 0)
            converted_logits = convert_gpt_idxs_to_bart(logits, lprobs.size(1), gpt2.bart_gpt2_dict)

            converted_logits = convert_gpt_idxs_to_bart(logits, lprobs.size(1), gpt2.bart_gpt2_dict)
            converted_logits = softmax(converted_logits)
            lprobs = lprobs * weights[0]
            concat_probs = lprobs.add(converted_logits * weights[1])

            if beam_width > 0:
                log_prob, indexes = torch.topk(concat_probs, beam_width)
            if top_p > 0.:
                concat_probs = concat_probs.add(n.block_penalty)
                sorted_logits, sorted_indices = torch.sort(concat_probs, descending=True)
                sigmoid_logs = 1 / (1 + torch.exp(-sorted_logits))
                sorted_indices_to_remove = sigmoid_logs > top_p

                print("sorted indicies to remove: ", sorted_indices_to_remove)
            node_penalty = n.block_penalty.clone()
            for new_k in range(beam_width):
                decoded_item = indexes[0][new_k].unsqueeze(0).unsqueeze(0)
                decoded_t = torch.cat((decoder_input, decoded_item), 1)
                if block_unigram is not None:
                    decoded_unique = decoded_t.unique(sorted=True)
                    if decoded_t.size(1) != decoded_unique.size(0):
                        decoded_unique_count = torch.stack([(decoded_t == d_u).sum() for d_u in decoded_unique])
                        idx_2_block = (torch.abs((block_unigram - decoded_unique_count)) < 0.0001).nonzero()
                        if idx_2_block.size(0) != 0:
                            for indxes in idx_2_block[0]:
                                idx_token = decoded_unique_count[indxes]
                                node_penalty[0][idx_token] -= 0.001
                log_p = log_prob[0][new_k].item()
                node = BeamSearchNode(n, decoded_t, n.logp + log_p, n.length + 1, node_penalty)
                score = -node.eval()
                nodes.put((score, node))

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(max_sentence_count)]

        beam_sentences = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            sentence = bart.model.decode(n.word_ids.squeeze(0))
            print("decoded sentence: ", sentence)
            beam_sentences.append(sentence)
        decoded_batch.append("#".join(beam_sentences))
    print("[DECODED BATCH]: ", decoded_batch)
    return decoded_batch

