import os
import sys

SEED = 5
JOIN_TOKEN = " "

# Paths
DATASETS_PATH = "datasets/"
SAVE_DATA_PATH = ".data/"

DATASETS_PATH = os.path.join(sys.path[0], DATASETS_PATH)
SAVE_DATA_PATH = os.path.join(sys.path[0], SAVE_DATA_PATH)

MODEL_PATH = "checkpoints/"
MODEL_PATH = os.path.join(sys.path[0], MODEL_PATH)

MODEL_SAVE_PATH = MODEL_PATH + 'seq2seq_model.pt'
MODEL_PREPROCESS_SAVE_PATH = MODEL_PATH + 'preprocess_model.pt'
MODEL_SAVE_FUNNY_PATH = MODEL_PATH + 'funny_model.pt'
MODEL_SAVE_POSITIVE_PATH = MODEL_PATH + 'positive_model.pt'
MODEL_SAVE_NEGATIVE_PATH = MODEL_PATH + 'negative_model.pt'
MODEL_SAVE_POETIC_PATH = MODEL_PATH + 'poetic_model.pt'
