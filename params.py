SEED = 5  # set seed value for deterministic results
CLIP = 10

N_EPOCHS = 15

JOIN_TOKEN = " "

# Preprocess
DATA_TYPE = "PERSONA_BOTH"  # TWITTER or PERSONA or PERSONA_BOTH
WITH_DESCRIPTION = True
CONTEXT_PAIR_COUNT = 0

WITH_ATTENTION = False
IS_BEAM_SEARCH = False

IS_TEST = True
CREATE_HISTOGRAM = False
PREPARE_DATA = False
PREPROCESS = False

MODEL_SAVE_PATH = 'seq2seq_model.pt'

config = {"train_batch_size": 10, "optimize_embeddings": False,
          "embedding_dim": 100, "hidden_dim": 512, "dropout_rate": 0.1, "num_layers": 1,
          "attention_model": 'concat'}
