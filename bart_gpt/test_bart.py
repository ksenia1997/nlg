import time

import torch
from fairseq.models.bart import BARTModel

from hub_interface import BartModel, GPT2Model
from hub_interface import sample, bart_gpt2_sample, create_idf, greedy_decoding

config = {
    "type_generation": "SPECIFICITY",  # (SPECIFICITY|COMBINE_MODELS|GREEDY_GPT2)
    "weight_BART": 0.3,
    "weight_GPT2": 0.7,
    "beam_width": 0,  # only beam or top_p must be set as non-zero
    "top_p": 0.5,  # threshold for Nucleus sampling
    "sample_num": 3,  # number of samples for Nucleus sampling
    "min_len": 3,  # min length of a generated sequence
    "max_len": 20,  # max length of a generated sequence| None if controlling length feature is used
    "max_sentence_count": 2,  # number of generated sequences
    "temperature": 1.,  # parameter for temperature sampling
    "unk_penalty": 0.001,  # penalty for unk in BART model
    "skip_ngram_number": 2,  # number of first words what would be generated only by BART model
    "block_unigram_counter": None,  # number of max repeating a word in a sequence
    "combine_number": 0,  # n for switching models
    "block_stop_words": False,  # feature of blocking stop words
    "length_feature": False  # feature of controlling length

}

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
bart = BARTModel.from_pretrained(
    'fairseq/checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='data_bin'
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 1

save_filename = "hypotheses/test_" + time.strftime('%d-%m-%Y_%H:%M:%S') + ".hypo"

with open('../.data/test.source') as source, open(save_filename, 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    if config["type_generation"] == "SPECIFICITY":
        idf_indexes = create_idf(bart)
    bart_model = BartModel(bart)
    gpt2 = GPT2Model()
    for sline in source:
        if config["type_generation"] == "GREEDY_GPT2":
            dec = greedy_decoding(gpt2, 10)
            fout.write(dec + '\n')
            fout.flush()
            continue
        if count % bsz == 0:
            with torch.no_grad():
                if config["type_generation"] == "COMBINE_MODELS":
                    weights = [config["weight_BART"], config["weight_GPT2"]]
                    hypotheses_batch = bart_gpt2_sample(bart_model, gpt2, weights, slines, config["beam_width"],
                                                        config["top_p"], config["sample_num"], config["min_len"],
                                                        config["max_len"],
                                                        config["max_sentence_count"], config["temperature"],
                                                        config["unk_penalty"], config["skip_ngram_number"],
                                                        config["block_unigram_counter"], config["combine_number"],
                                                        config["block_stop_words"], config["length_feature"])
                if config["type_generation"] == "SPECIFICITY":
                    hypotheses_batch = sample(bart, idf_indexes, slines, beam=3, lenpen=2.0, max_len_b=200, min_len=5,
                                              no_repeat_ngram_size=2)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
                if count % bsz == 0:
                    print("H: ", hypothesis)
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines:
        hypotheses_batch = sample(bart, idf_indexes, slines, beam=3, lenpen=2.0, max_len_b=100, min_len=5,
                                  no_repeat_ngram_size=2)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()
