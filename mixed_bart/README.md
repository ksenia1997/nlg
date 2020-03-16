
#Download and Install fairseq 
```
git clone https://github.com/pytorch/fairseq.git
cd fairseq
pip install --editable .
```

#Download BART model
```
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar -xzvf bart.large.tar.gz
```

#Fine-tuning BART on the dataset
```
#create a directory in /bart
mkdir data-bin

bash BPE_preprocess.sh
bash binarize_dataset.sh
bash fine_tune.sh

python3 test_bart.py
```