fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "../datasets/train.bpe" \
  --validpref "../datasets/val.bpe" \
  --destdir "data-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;