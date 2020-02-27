SEED = 5  # set seed value for deterministic results
CLIP = 10

N_EPOCHS = 10

JOIN_TOKEN = " "

# Paths
DATA_PATH = "./datasets/"
MODEL_PATH = "./models/"

# Create special type of data for BART tune
PREPARE_BART = False
# Preprocess
DATA_TYPE = "PERSONA_BOTH"  # TWITTER or PERSONA or PERSONA_BOTH (PERSONA_BOTH contains descriptions of both people)
WITH_DESCRIPTION = True
CONTEXT_PAIR_COUNT = 0

WITH_ATTENTION = True
IS_BEAM_SEARCH = False

CREATE_HISTOGRAM = False
PREPARE_DATA = True
PREPROCESS = False
IS_TEST = False

MODEL_SAVE_PATH = MODEL_PATH + 'seq2seq_model.pt'
MODEL_PREPROCESS_SAVE_PATH = MODEL_PATH + 'preprocess_model.pt'

config = {"train_batch_size": 32, "optimize_embeddings": False,
          "embedding_dim": 300, "hidden_dim": 512, "dropout_rate": 0.1, "num_layers": 2,
          "attention_model": 'concat', "transformer_d_model": 512, "transformer_heads": 8, "transformer_n": 6,
          "transformer_dropout": 0.1}
