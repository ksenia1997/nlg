import copy
import logging
import math
import numpy as np
import pickle
from typing import List
import json

import torch
from fairseq import search
from sequence_generator import EnsembleModel
from sequence_generator import SequenceGenerator
import gpt.src.encoder as gpt2_encoder
import gpt.src.model as gpt2_model
import tensorflow as tf

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
    max_nidf  = torch.max(matrix_tf_idf)
    min_nidf = torch.min(matrix_tf_idf)
    matrix_tf_idf = torch.log(matrix_tf_idf) - min_nidf
    matrix_tf_idf = matrix_tf_idf / (max_nidf - min_nidf)
    return matrix_tf_idf


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


def bart_beam_decode(model, tf_idf, input_tokens, beam_width, max_len, temperature, unk_penalty):
    decoded_batch = []
    max_len = max(max_len, 2)
    beam_width = max(beam_width, 2)
    sentences = ""
    input_tokens = [model.encode(sentence) for sentence in input_tokens]
    sample = model._build_sample(input_tokens)

    pad = model.task.target_dictionary.pad()
    unk = model.task.target_dictionary.unk()
    eos = model.task.target_dictionary.eos()

    encoder_input = {
        k: v for k, v in sample['net_input'].items()
        if k != 'prev_output_tokens'
    }

    src_tokens = encoder_input['src_tokens']
    src_lengths = (src_tokens.ne(eos) & src_tokens.ne(pad)).long().sum(dim=1)
    input_size = src_tokens.size()
    print("Input size: ", input_size)
    bsz = input_size[0]
    src_len = input_size[1]
    with torch.no_grad():
        ensemble_model = EnsembleModel([model.model])
        encoder_outs = ensemble_model.forward_encoder(encoder_input)
        print("encoder outs: ", encoder_outs)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_width).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = ensemble_model.reorder_encoder_out(encoder_outs, new_order)
        print("encoder outs reordered: ", encoder_outs)

        tokens = src_tokens.new(bsz * beam_width, max_len + 2).long().fill_(pad)
        print("tokens: ", tokens.size(), tokens)
        tokens[:, 0] = eos
        print("tokens with eos: ", tokens)

        lprobs, avg_attn_scores = ensemble_model.forward_decoder(
            tokens[:, :1], encoder_outs, temperature=temperature,
        )
        #lprobs = lprobs.add(tf_idf)
        lprobs[:, pad] = -math.inf  # never select pad
        lprobs[:, unk] -= unk_penalty  # apply unk penalty

        enc_gpt2 = gpt2_encoder.get_encoder('117M')
        hparams = gpt2_model.default_hparams()
        with open(os.path.join('gpt/src/models', '117M', 'hparams.json')) as f:
             hparams.override_from_dict(json.load(f))   
        print("enc gpt2: ", enc_gpt2)
        print("hparams: ", hparams)
        length = hparams.n_ctx
        print("length: ", length)
        start_token = enc_gpt2.encoder['<|endoftext|>']
        print("start token: ", start_token)
        context = tf.fill([1, 1], start_token)
        print("context: ", context)
        print("n vocab: ", hparams.n_vocab)
        lm_output = gpt2_model.model(hparams=hparams, X=context, past=None, reuse=tf.AUTO_REUSE)
        print("lm output: ", lm_output)
        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            logits = logits.eval()
        print("LOGITS: ", logits, type(logits)) 
        logits = torch.tensor(logits).float()
        print("logits torch: ", logits)

