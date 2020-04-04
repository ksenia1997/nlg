from preprocessing import *

config = {  # TWITTER|PERSONA|PERSONA_BOTH (PERSONA_BOTH contains descriptions of both people)
    "dataset_type_seq2seq": 'PERSONA_BOTH',
    "with_description": True,  # In case PERSONA (with/without persona description)
    # In case of truncating PERSONA data. 0 if it is not needed to be truncated, N if you need to reduce dialogue.
    "context_pair_count": 0,
    "dataset_decoding": 'JOKE',  # JOKE|POETIC (only for text generation)
    "model_type": 'BART',  # BART|GPT2|Basemodel|Decoding data should be saved in a special format
    "control_attribute": 'data_for_idf'  # data_for_idf|prepare_dict
}

if __name__ == "__main__":
    prepare_data(config)
