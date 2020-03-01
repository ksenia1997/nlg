from seq2seq import run_seq2seq

config = {"train_batch_size": 32,
          "embedding_dim": 300,
          "hidden_dim": 512,
          "dropout_rate": 0.1,
          "num_layers": 2,
          "n_epochs": 10,
          "clip": 10,
          "attention_model": 'concat',
          "optimize_embeddings": False,
          "transformer_d_model": 512,
          "transformer_heads": 8,
          "transformer_n": 6,
          "transformer_dropout": 0.1}

if __name__ == "__main__":
    run_seq2seq(config)
