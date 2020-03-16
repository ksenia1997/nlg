fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "../.data/train.bpe" \
  --validpref "../.data/val.bpe" \
  --destdir "data-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;