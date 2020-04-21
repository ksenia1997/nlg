from preprocessing import *

config = {  # PERSONA|PERSONA_BOTH (PERSONA_BOTH contains descriptions of both people)
    "dataset_type_seq2seq": 'PERSONA_BOTH',
    "pretraining_dataset": "TWITTER",
    "with_description": True,  # In case PERSONA (with/without persona description)
    # In case of truncating PERSONA data. 0 if it is not needed to be truncated, N if you need to reduce dialogue.
    "context_pair_count": 0,
    "model_seq2seq_type": 'BART'  # BART|Basemodel
}

if __name__ == "__main__":
    prepare_seq2seq_data(config)
