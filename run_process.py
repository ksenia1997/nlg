from processes import run_model

config = {"train_batch_size": 32,
          "embedding_dim": 300,
          "hidden_dim": 512,
          "dropout_rate": 0.1,
          "num_layers": 2,
          "n_epochs": 10,
          "clip": 10,
          "teacher_forcing_ratio": 0.1,
          "with_attention": False,  # only for seq2seq model
          "attention_model": 'concat',  # dot|general|concat
          "decoding_type": 'beam',  # beam|greedy|weighted_beam
          "beam_width": 4,
          "max_sentences": 3,  # number of generated sentences by beam decoding
          "max_sentence_len": 40,  # max length of a sentence generated by beam decoding
          # TWITTER|PERSONA|PERSONA_BOTH|JOKE|POETIC (PERSONA_BOTH contains descriptions of both people)
          "data_type": 'POETIC',
          "with_description": True,  # In case PERSONA (with/without persona description)
          # In case of truncating PERSONA data. 0 if it is not needed to be truncated, N if you need to reduce dialogue.
          "context_pair_count": 0,
          "data_BART": False,  # data for BART model should be saved in a special format
          "data_GPT2": True,  # data for GPT2 model should be saved in a special format
          "data_for_idf": False,
          "prepare_data": True,
          "prepare_dict": False,
          "train_preprocess": False,  # to train preprocess, it's necessary to prepare TWITTER data before
          "with_preprocess": False,  # to train a model with preprocessed model
          "process": 'test',  # train|test|train_lm
          "is_stylized_generation": True,  # while testing generate text with different styles
          "with_stylized_lm": False,  # if "is_stylized_generation" is True
          "with_controlling_attributes": True,  # if "is_stylized_generation" is True
          "style": "funny"}

if __name__ == "__main__":
    run_model(config)
