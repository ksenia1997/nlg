import torch
from fairseq.models.bart import BARTModel

from hub_interface import sample, create_tf_idf, bart_beam_decode

bart = BARTModel.from_pretrained(
    'fairseq/checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='data_bin'
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 32
with open('../.data/test.source') as source, open('test.hypo', 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    #tf_idf_indexes = create_tf_idf(bart)
    tf_idf_indexes = None
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                print("before bart decode")
                bart_beam_decode(bart, tf_idf_indexes, slines, 3, 100, 1, 0.001)
                print("after bart decode")
                hypotheses_batch = sample(bart, tf_idf_indexes, slines, beam=3, lenpen=2.0, max_len_b=100, min_len=5,
                                          no_repeat_ngram_size=2)
             
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
                if count % bsz*3 == 0:
                    print("H: ",hypothesis)
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = sample(bart, tf_idf_indexes, slines, beam=3, lenpen=2.0, max_len_b=100, min_len=5, no_repeat_ngram_size=2)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()
