# Download and Install fairseq 
```
# In the directory nlg/bart_gpt
git clone https://github.com/pytorch/fairseq.git
cd fairseq
pip install --editable .

export PYTHONPATH=path/to/fairseq
```

# Download BART model
```
# In the directory nlg/bart_gpt
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar -xzvf bart.large.tar.gz
```
# Prepare data for BART
```
# create a directory in nlg/bart_gpt
mkdir data_bin

bash BPE_preprocess.sh
bash binarize_dataset.sh
```
# Fine-tuning BART on the dataset
```
bash fine_tune.sh
# In the directory nlg/
export PYTHONPATH=./bart_gpt/fairseq:$PYTHONPATH
python3 ./bart/test_bart.py
```


# GPT-2 model
```
git clone https://github.com/Tenoke/gpt-2.git
mv gpt-2 gpt
python3 gpt/download_model.py 117M
export PYTHONPATH=/path/to/gpt/src

```

# Fine-tuning GPT-2 on a stylistic dataset
```
python ./encode.py <file|directory|glob> /path/to/encoded.npz
python ./train.py --dataset /path/to/encoded.npz

```

# GPT-2 checkpoint
```
mkdir checkpoint_gpt
cp -R /path/to/generated/checkpoints ./checkpoint_gpt/
```