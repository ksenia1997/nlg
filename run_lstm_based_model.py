from processes import run_train_lm_model

config = {"train_batch_size": 32,
          "embedding_dim": 300,
          "hidden_dim": 512,
          "dropout_rate": 0.1,
          "num_layers": 2,
          "n_epochs": 10,
          "clip": 10,
          "teacher_forcing_ratio": 0.5,
          "style": "funny"  # to train LM model styles: funny|poetic|positive|negative
          }

if __name__ == "__main__":
    run_train_lm_model(config)
