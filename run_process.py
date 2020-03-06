from processes import run_model

config = {"train_batch_size": 32,
          "embedding_dim": 300,
          "hidden_dim": 512,
          "dropout_rate": 0.1,
          "num_layers": 2,
          "n_epochs": 10,
          "clip": 10,
          "teacher_forcing_ratio": 0.1,
          "with_attention": True,  # only for seq2seq model
          "attention_model": 'concat',  # dot|general|concat
          "decoding_type": 'weighted_beam',  # beam|greedy|weighted_beam
          # TWITTER|PERSONA|PERSONA_BOTH|JOKE (PERSONA_BOTH contains descriptions of both people)
          "data_type": 'JOKE',
          "with_description": True,  # In case PERSONA (with/without persona description)
          # In case of truncating PERSONA data. 0 if it is not needed to be truncated, N if you need to reduce dialogue.
          "context_pair_count": 0,
          "data_BART": False,  # data for BART model should be saved in a special format
          "prepare_data": False,
          "prepare_dict": False,
          "train_preprocess": False,  # to train preprocess, prepare TWITTER data
          "with_preprocess": False,
          "process": 'test',  # train|test|train_lm
          "is_stylized_generation": True,  # while testing generate text with different styles
          "style": "funny",
          "optimize_embeddings": False,
          "transformer_d_model": 512,
          "transformer_heads": 8,
          "transformer_n": 6,
          "transformer_dropout": 0.1}

if __name__ == "__main__":
    run_model(config)
