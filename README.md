```
sudo snap install docker

# to train seq2seq model
docker build  -f Dockerfile.train_baseline_seq2seq -t seq2seq .
docker run seq2seq

# to train lm model
docker build  -f Dockerfile.train_baseline_lm -t lm .
docker run lm
```

 

