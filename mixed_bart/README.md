
#Download and Install fairseq 
```
# In the directory nlg/bart
git clone https://github.com/pytorch/fairseq.git
cd fairseq
pip install --editable .

export PYTHONPATH=path/to/fairseq
```

#Download BART model
```
# In the directory nlg/bart
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar -xzvf bart.large.tar.gz
```

#Fine-tuning BART on the dataset
```
# create a directory in nlg/bart
mkdir data-bin

bash BPE_preprocess.sh
bash binarize_dataset.sh
bash fine_tune.sh

# In the directory nlg/
export PYTHONPATH=./bart/fairseq:$PYTHONPATH
python3 ./bart/test_bart.py
```


#GPT-2 model
```angular2
git clone https://github.com/Tenoke/gpt-2.git
in src/encoder.py change paths

```