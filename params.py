SEED = 5
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

WITH_ATTENTION = False
IS_BEAM_SEARCH = True

CREATE_HISTOGRAM = False
PREPARE_DATA = False
TRAIN_PREPROCESS = False
PREPROCESS = False
IS_TEST = True

MODEL_SAVE_PATH = MODEL_PATH + 'seq2seq_model.pt'
MODEL_PREPROCESS_SAVE_PATH = MODEL_PATH + 'preprocess_model.pt'


