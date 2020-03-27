
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
mkdir data_bin

bash BPE_preprocess.sh
bash binarize_dataset.sh
bash fine_tune.sh

# In the directory nlg/
export PYTHONPATH=./bart/fairseq:$PYTHONPATH
python3 ./bart/test_bart.py
```


#GPT-2 model
```
git clone https://github.com/Tenoke/gpt-2.git
export PYTHONPATH=/path/to/gpt-2/src
python encode.py /path/to/your_data.txt /path/to/encoded.npz
in src/encoder.py change paths

```