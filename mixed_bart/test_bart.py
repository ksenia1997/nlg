import torch
from fairseq.models.bart import BARTModel
from hub_interface import BartModel, GPT2Model
from hub_interface import sample, bart_beam_decode, create_idf, greedy_decoding

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
with open('../.data/test.source') as source, open('test_poeticc19.hypo', 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    if SPECIFICITY:
        idf_indexes = create_idf(bart)
    bart_model = BartModel(bart)
    gpt2 = GPT2Model()
    for sline in source:
        greedy_decoding(bart_model, gpt2, 20)
        continue
        if count % bsz == 0:
            with torch.no_grad():
                if COMBINE_MODELS:
                    hypotheses_batch = bart_beam_decode(bart_model, gpt2, [0.3, 0.7], slines, beam_width=3, top_p=0,
                                                        min_len=3, max_len=20, max_sentence_count=2, temperature=1,
                                                        unk_penalty=0.001)
                if SPECIFICITY:
                    hypotheses_batch = sample(bart, idf_indexes, slines, beam=3, lenpen=2.0, max_len_b=100, min_len=5,
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
        hypotheses_batch = sample(bart, tf_idf_indexes, slines, beam=3, lenpen=2.0, max_len_b=100, min_len=5,
                                  no_repeat_ngram_size=2)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()

