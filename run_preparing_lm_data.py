from preprocessing import *

config = {
    "model_lm_type": 'LSTM',  # GPT2|LSTM
    "feature_based_modifications": False
}

if __name__ == "__main__":
    if config["feature_based_modifications"]:
        prepare_decoding_feature_modifications()
    prepare_lm_data(config)
