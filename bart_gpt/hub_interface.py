import copy
import json
import logging
import math
import operator
import os
import pickle
import random
from queue import PriorityQueue
from typing import List

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from fairseq import search

import gpt.src.encoder as gpt2_encoder
import gpt.src.model as gpt2_model
from gpt.src.sample import sample_sequence, top_k_logits
from sequence_generator import EnsembleModel
from sequence_generator import SequenceGenerator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
              'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
              'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
              'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me',
              'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both',
              'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and',
              'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over',
              'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too',
              'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my',
              'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', '.', ',', '!', '?']


def create_idf(model):
    with open("../datasets/tf-idf", "rb") as fp:
        sentences = pickle.load(fp)
    indexes = list(range(0, len(model.task.target_dictionary)))
    sentences_length = len(sentences)
    indexes_dict = dict(zip(indexes, [1 for x in range(0, len(indexes))]))
    for sentence in sentences:
        enc_sentence = model.encode(sentence)
        unique_tokens = np.unique(np.array(enc_sentence))
        for token in unique_tokens:
            indexes_dict[token] += 1
    matrix_idf = torch.arange(len(model.task.target_dictionary)).type(torch.FloatTensor)
    for k, v in indexes_dict.items():
        matrix_idf[k] = sentences_length / v
    max_nidf = torch.max(matrix_idf)
    min_nidf = torch.min(matrix_idf)
    matrix_idf = torch.log(matrix_idf) - min_nidf
    matrix_idf = matrix_idf / (max_nidf - min_nidf)
    return matrix_idf


def create_tf_idf(model, filename):
    f = open(filename, 'r')
    indexes = list(range(0, len(model.task.target_dictionary)))
    indexes_idf_dict = dict(zip(indexes, [1 for x in range(0, len(indexes))]))
    indexes_tf_dict = dict(zip(indexes, [1 for x in range(0, len(indexes))]))
    word_counter = 0
    doc_counter = 0
    for line in f:
        if line == "":
            continue
        doc_counter += 1
        separate_line = line.split()
        doc = ""
        for word in separate_line:
            if word not in stop_words:
                word_counter += 1
                doc += word + " "
        enc_sentences = model.encode(doc)
        enc_sentences = np.array(enc_sentences)
        for token in enc_sentences:
            indexes_tf_dict[token] += 1
        unique_tokens = np.unique(enc_sentences)
        for token in unique_tokens:
            indexes_idf_dict[token] += 1
    matrix_tf_idf = torch.arange(len(model.task.target_dictionary)).type(torch.FloatTensor)
    for k, v in indexes_idf_dict.items():
        tf = indexes_tf_dict[k] / word_counter
        idf = np.log(doc_counter / v)
        matrix_tf_idf[k] = tf * idf
    return matrix_tf_idf * 10


def sample(model, idf, sentences: List[str], beam: int = 1, verbose: bool = False,
           **kwargs) -> str:
    input = [model.encode(sentence) for sentence in sentences]
    hypos = generate(model, idf, input, beam, verbose, **kwargs)
    return [model.decode(x['tokens']) for x in hypos]


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
            converted_logp.append(-math.inf)
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
    def __init__(self, previous_node, word_ids, log_prob, length, penalty, skip_ngram_number, prev_context_gpt2,
                 max_len):
        self.prev_node = previous_node
        self.word_ids = word_ids
        self.logp = log_prob
        self.length = length
        self.block_penalty = penalty
        self.skip_n = skip_ngram_number
        self.prev_context_gpt2 = prev_context_gpt2
        if max_len is None:
            self.max_len = random.randrange(15, 50)
        else:
            self.max_len = max_len

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
        self.checkpoint_path = 'checkpoint/run1'
        with open(os.path.join('models', '117M', 'hparams.json')) as f:
            self.hyper_params.override_from_dict(json.load(f))


def greedy_decoding(gpt2: GPT2Model, max_len=100):
    start_token = gpt2.encoder.encoder[gpt2.eos]
    with tf.Session(graph=tf.Graph()) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        context = tf.fill([1, 1], start_token)
        output = context
        context_output = gpt2_model.model(hparams=gpt2.hyper_params, X=context[:, :-1], past=None,
                                          reuse=tf.AUTO_REUSE)
        past = context_output['present']
        for i in range(max_len):
            lm_output = gpt2_model.model(hparams=gpt2.hyper_params, X=context, past=past, reuse=tf.AUTO_REUSE)
            logits = lm_output['logits'][:, :, :gpt2.hyper_params.n_vocab]
            logits = logits[:, -1, :]
            past = tf.concat([past, lm_output['present']], axis=-2)
            logits = top_k_logits(logits, k=3)
            context = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            output = tf.concat([output, context], axis=1)
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(gpt2.checkpoint_path)
            saver.restore(sess, ckpt)
        out = sess.run(output)
    decoded_gpt2 = gpt2.encoder.decode(out[0])
    return decoded_gpt2


def gpt_sample(gpt2: GPT2Model, seed=None, top_k=3, temperature=1, batch_size=2, length=20):
    with tf.Session(graph=tf.Graph()) as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample_sequence(
            hparams=gpt2.hyper_params, length=length,
            start_token=gpt2.encoder.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature, top_k=top_k)[:, 1:]
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(gpt2.checkpoint_path)
        saver.restore(sess, ckpt)
        out = sess.run(output)
        for i in range(batch_size):
            text = gpt2.encoder.decode(out[i])
            print(text)


def bart_gpt2_sample(bart: BartModel, gpt2: GPT2Model, weights, input_tokens, beam_width: int = 0, top_p: float = 0.0,
                     min_len: int = 3, max_len: int = None, max_sentence_count: int = 2, temperature: float = 1.,
                     unk_penalty: float = 0.001, skip_ngram_number: int = 1, block_unigram_counter: int = None,
                     combine_number: int = 0, block_stop_words: bool = False):
    '''

    Args:
        bart: BART model
        gpt2: GPT2 model
        weights: weights for weighted decoding
        input_tokens: array of inputs
        beam_width: parameter for Beam Search
        top_p: parameter for Nucleus Sampling
        min_len:
        max_len:
        max_sentence_count: number of sentences generated for 1 input
        temperature:
        unk_penalty:
        skip_ngram_number:
        block_unigram_counter:
        combine_number:
        block_stop_words:

    Returns: array of generated hypotheses

    '''

    assert (max_len is None or max_len > 2)
    assert max_sentence_count > 1
    assert min_len > 1, "Minimal length of a generated utterance should be bigger than 1"
    assert bool(beam_width) != bool(top_p), "Set beam width or top p"

    decoded_batch = []
    log_softmax = torch.nn.LogSoftmax(dim=1)
    MAX_TOP_P_DIM = 50

    with tf.Session(graph=tf.Graph()) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(len(input_tokens)):
            endnodes = []
            nodes = PriorityQueue()
            print("input_token: ", input_tokens[i])
            # BART
            bart.ensemble_model = EnsembleModel([bart.model.model])
            enc_tokens = [bart.model.encode(input_tokens[i])]
            sample = bart.model._build_sample(enc_tokens)
            encoder_input = {
                k: v for k, v in sample['net_input'].items()
                if k != 'prev_output_tokens'
            }
            src_tokens = encoder_input['src_tokens']
            encoder_outs = bart.ensemble_model.forward_encoder(encoder_input)
            target_tokens = src_tokens.new(1, 1).long().fill_(bart.eos)
            block_penalty = torch.zeros([1, 50264], dtype=torch.float64).to(device)
            # GPT2
            start_gpt = gpt2.encoder.encoder[gpt2.eos]
            context = tf.fill([1, 1], start_gpt)
            context_out = gpt2_model.model(hparams=gpt2.hyper_params, X=context[:, :-1], past=None, reuse=tf.AUTO_REUSE)

            node = BeamSearchNode(None, target_tokens, 0, 1, block_penalty, skip_ngram_number - 1,
                                  context_out['present'], max_len)
            nodes.put((-node.eval(), node))
            counter = 0
            while True:
                score, n = nodes.get()
                decoder_input = n.word_ids
                if (n.word_ids[0][-1].item() == bart.eos and n.prev_node is not None) or n.length >= n.max_len:
                    print("max len: ", n.max_len)
                    print("n len: ", n.length)
                    print("last item: ", n.word_ids[0][-1].item())
                    if n.length < min_len:
                        continue
                    endnodes.append((score, n))
                    if len(endnodes) >= max_sentence_count:
                        break
                    else:
                        continue
                lprobs_bart, avg_attn_scores = bart.ensemble_model.forward_decoder(
                    decoder_input, encoder_outs, temperature=temperature)
                lprobs_bart[:, bart.pad] = -math.inf  # never select pad
                lprobs_bart[:, bart.unk] -= unk_penalty  # apply unk penalty
                concat_probs = lprobs_bart * weights[0]

                if n.skip_n > 0:
                    print("skip")
                    n.skip_n -= 1
                else:
                    gpt_item = gpt2.bart_gpt2_dict[decoder_input[0][-1]]
                    if gpt_item == 50257:
                        gpt_item = 0
                    context = tf.convert_to_tensor([[gpt_item]])
                    lm_output = gpt2_model.model(hparams=gpt2.hyper_params, X=context, past=n.prev_context_gpt2,
                                                 reuse=tf.AUTO_REUSE)
                    logits = lm_output['logits'][:, :, :gpt2.hyper_params.n_vocab]
                    logits = logits[:, -1, :]
                    n.prev_context_gpt2 = tf.concat([n.prev_context_gpt2, lm_output['present']], axis=-2)

                    saver = tf.train.Saver()
                    ckpt = tf.train.latest_checkpoint(gpt2.checkpoint_path)
                    saver.restore(sess, ckpt)

                    # converting tf.Tensor to torch.tensor
                    logits = logits.eval()
                    logits = torch.tensor(logits)

                    converted_logits = convert_gpt_idxs_to_bart(logits, lprobs_bart.size(1), gpt2.bart_gpt2_dict)
                    lprobs_gpt = log_softmax(converted_logits)
                    print("gpt added")
                    concat_probs = concat_probs.add(lprobs_gpt * weights[1])

                node_penalty = n.block_penalty.clone()
                counter += 1
                if counter < combine_number:
                    print("combine number")
                    concat_probs = lprobs_bart
                elif counter < 2 * combine_number:
                    print("combine number2")
                    concat_probs = lprobs_gpt
                else:
                    counter = 0
                if beam_width > 0:
                    log_prob, indexes = torch.topk(concat_probs, beam_width)
                if top_p > 0.:
                    sorted_logits, sorted_indices = torch.sort(concat_probs, descending=True)
                    sigmoid_logs = F.softmax(sorted_logits, dim=1)
                    cum_sum = torch.cumsum(sigmoid_logs, 1)
                    logits_top_p = cum_sum < top_p
                    indexes = sorted_indices[logits_top_p].unsqueeze(0)
                    log_prob = sorted_logits[logits_top_p].unsqueeze(0)
                    indexes = indexes[0][:MAX_TOP_P_DIM].unsqueeze(0)
                    log_prob = log_prob[0][:MAX_TOP_P_DIM].unsqueeze(0)
                if block_stop_words:
                    isFirst = True
                    for k in range(indexes.size(1)):
                        word = bart.model.decode(indexes[0][k].unsqueeze(0))
                        if word not in stop_words:
                            if isFirst:
                                new_indexes = indexes[0][k]
                                new_logp = log_prob[0][k]
                                isFirst = False
                            else:
                                new_indexes = torch.cat((new_indexes, indexes[0][k]), 1)
                                new_logp = torch.cat((new_logp, log_prob[0][k]), 1)
                    if not isFirst:
                        indexes = new_indexes
                        log_prob = new_logp
                for new_k in range(indexes.size(1)):
                    decoded_item = indexes[0][new_k].unsqueeze(0).unsqueeze(0)
                    decoded_t = torch.cat((decoder_input, decoded_item), 1)
                    if block_unigram_counter is not None:
                        decoded_unique = decoded_t.unique(sorted=True)
                        if decoded_t.size(1) != decoded_unique.size(0):
                            decoded_unique_count = torch.stack([(decoded_t == d_u).sum() for d_u in decoded_unique])
                            idx_2_block = (torch.abs((block_unigram_counter - decoded_unique_count)) < 0.0001).nonzero()
                            if idx_2_block.size(0) != 0:
                                for indxes in idx_2_block[0]:
                                    idx_token = decoded_unique_count[indxes]
                                    node_penalty[0][idx_token] -= 0.01
                    log_p = log_prob[0][new_k].item()
                    node = BeamSearchNode(n, decoded_t, n.logp + log_p, n.length + 1, node_penalty, n.skip_n,
                                          n.prev_context_gpt2, max_len)
                    score = -node.eval()
                    nodes.put((score, node))

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(max_sentence_count)]

            beam_sentences = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                sentence = bart.model.decode(n.word_ids.squeeze(0))
                sentence = sentence.replace('\n', ' ')
                print("decoded sentence: ", sentence)
                beam_sentences.append(sentence)
            decoded_batch.append(" # ".join(beam_sentences))
    print("[DECODED BATCH]: ", decoded_batch)
    return decoded_batch
