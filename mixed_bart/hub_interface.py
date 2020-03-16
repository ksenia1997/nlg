import copy
import logging
from typing import List

import torch
from fairseq import search
from sequence_generator import SequenceGenerator

logger = logging.getLogger(__name__)


def sample(model, sentences: List[str], beam: int = 1, verbose: bool = False, **kwargs) -> str:
    input = [model.encode(sentence) for sentence in sentences]
    hypos = generate(model, input, beam, verbose, **kwargs)
    return [model.decode(x['tokens']) for x in hypos]


def generate(model, tokens: List[torch.LongTensor], beam: int = 5, verbose: bool = False, **kwargs) -> torch.LongTensor:
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

    def getarg(name, default):
        return getattr(gen_args, name, getattr(model.args, name, default))

    # Process top predictions
    hypos = [x[0] for x in translations]
    hypos = [v for _, v in sorted(zip(sample['id'].tolist(), hypos))]
    return hypos

