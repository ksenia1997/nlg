# Prepare data
```
./nlg/run_preparing_seq2seq_data.py # set 'model_seq2seq_type' to 'BART'
./nlg/run_preparing_lm_data.py # set 'model_lm_type' to 'GPT2'
```

# Requirements 
```
pip install -r requirements.txt
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

```
# BART
Anaconda is necessary! CUDA version 10.1
### Download and Install fairseq 
```
# In the directory nlg/bart_gpt
git clone https://github.com/pytorch/fairseq.git
cd fairseq
pip install --editable .

export PYTHONPATH=path/to/fairseq
```

### Download BART model
```
# In the directory nlg/bart_gpt
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar -xzvf bart.large.tar.gz
```
### Prepare data for BART
```
# create a directory in nlg/bart_gpt
mkdir data_bin

bash BPE_preprocess.sh
bash binarize_dataset.sh
```
### Fine-tuning BART on the dataset
```
bash fine_tune.sh
# In the directory nlg/
export PYTHONPATH=nlg/bart_gpt/fairseq:$PYTHONPATH
python3 nlg/bart_gpt/test_bart.py
```


# GPT-2 model
### Download GPT-2 model
```
git clone https://github.com/Tenoke/gpt-2.git
mv gpt-2 gpt
python3 gpt/download_model.py 117M
export PYTHONPATH=/path/to/gpt/src

```

### Fine-tuning GPT-2 on a stylistic dataset
```
python ./encode.py <file|directory|glob> /path/to/encoded.npz
python ./train.py --dataset /path/to/encoded.npz

```

### GPT-2 checkpoint
```
mkdir nlg/bart_gpt/checkpoint_gpt
cp -R /path/to/generated/checkpoints ./checkpoint_gpt/
```
