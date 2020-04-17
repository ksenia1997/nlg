import torch
from fairseq.models.bart import BARTModel

from hub_interface import BartModel, GPT2Model
from hub_interface import sample, bart_gpt2_sample, create_idf, greedy_decoding, create_tf_idf

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

SPECIFICITY = False
COMBINE_MODELS = True
GREEDY_GPT2 = False
with open('../.data/test.source') as source, open('hypotheses/beam_sst_pos55_n3_models.hypo', 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    if SPECIFICITY:
        #idf_indexes = None
        idf_indexes = create_idf(bart)
        #idf_indexes = create_tf_idf(bart, "../datasets/sst_positive_sentences.txt")
    bart_model = BartModel(bart)
    gpt2 = GPT2Model()
    for sline in source:
        if GREEDY_GPT2:
            # gpt_sample(gpt2)
            # exit()
            dec = greedy_decoding(gpt2, 10)
            fout.write(dec + '\n')
            fout.flush()
            continue
        if count % bsz == 0:
            with torch.no_grad():
                if COMBINE_MODELS:
                    hypotheses_batch = bart_gpt2_sample(bart_model, gpt2, [0.5, 0.5], slines, beam_width=20, top_p=0.,
                                                        min_len=3, max_len=40, max_sentence_count=4, skip_ngram_number=3)
                if SPECIFICITY:
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
    if slines != []:
        hypotheses_batch = sample(bart, idf_indexes, slines, beam=3, lenpen=2.0, max_len_b=100, min_len=5,
                                  no_repeat_ngram_size=2)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()
