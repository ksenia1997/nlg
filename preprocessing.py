import pickle
import random
import re
import string

import en_core_web_sm
import nltk
import spacy

nltk.download('stopwords')
from nltk.corpus import stopwords
from spacy.tokenizer import Tokenizer

from params import *
from utils.csv_process import *
from utils.json_process import *
from utils.save_model_data import save_data_for_BART, save_data_for_GPT2


def personas_description(line):
    your_persona = re.findall(r"your persona:(.*)", line)
    partner_persona = re.findall(r"partner's persona:(.*)", line)
    if your_persona:
        return True, your_persona[0], ""
    elif partner_persona:
        return True, "", partner_persona[0]
    else:
        return False, "", ""


def prepare_both_Persona_chat(nlp, filename):
    with open(filename) as fp:
        your_persona_description = []
        partner_persona_description = []
        counter = 0
        delimiter_context_dialogue = " CC "
        delimiter = " # "
        delimiter_start = " SS "

        person1, person2 = [], []
        add_description = True
        dialog_counter = 0
        train_data, valid_data, test_data = [], [], []

        for line in fp:
            counter += 1
            is_description, y, p = personas_description(line)
            if is_description:
                add_description = True
                if y != "":
                    your_persona_description.append(y)
                if p != "":
                    partner_persona_description.append(p)
            else:
                if add_description:
                    dialog_counter += 1
                    add_description = False
                    context = ""
                    data = []
                    for i in range(len(person1)):
                        partner_persona_desc_str = JOIN_TOKEN.join(tokenize(partner_persona_desc_str, nlp)[1])
                        your_persona_desc_str = JOIN_TOKEN.join(tokenize(your_persona_desc_str, nlp)[1])
                        data.append(partner_persona_desc_str + delimiter_context_dialogue + delimiter_start + context)
                        data.append(person1[i])
                        context += person1[i] + delimiter
                        data.append(your_persona_desc_str + delimiter_context_dialogue + delimiter_start + context)
                        data.append(person2[i])
                        context += person2[i] + delimiter
                    your_persona_desc_str = delimiter.join(your_persona_description)
                    partner_persona_desc_str = delimiter.join(partner_persona_description)
                    person1, person2 = [], []
                    your_persona_description = []
                    partner_persona_description = []
                    if counter % 5 == 0:
                        valid_data += data
                    elif counter % 9 == 0:
                        test_data += data
                    else:
                        train_data += data
            sentences = line.split("\t")
            if len(sentences) > 1:
                utterance1 = re.findall(r"\d+ (.*)", sentences[0])[0]
                utterance2 = sentences[1]
                utterance1 = JOIN_TOKEN.join(tokenize(utterance1, nlp)[1])
                utterance2 = JOIN_TOKEN.join(tokenize(utterance2, nlp)[1])
                person1.append(utterance1)
                person2.append(utterance2)

    return train_data, valid_data, test_data


def prepare_Twitter_data(nlp, filename):
    train_data = []
    valid_data = []
    test_data = []
    counter = 0
    with open(filename) as fp:
        for line in fp:
            line = JOIN_TOKEN.join(tokenize(line, nlp)[1])
            train_data.append(line)
            if counter % 10 == 0:
                valid_data.append(line)
            if counter % 20 == 0:
                test_data.append(line)
            counter += 1
    return train_data, valid_data, test_data


def prepare_Persona_chat(nlp, filename, context_pair_count, with_description):
    """

    Args:
        nlp: tokenizer
        filename: the name of the file
        context_pair_count: number of the dialogue pair for the truncation
        with_description: bool value, which indicates if the creating dataset will be with the persona description or not

    Returns:

    """
    train_data = []
    test_data = []
    valid_data = []
    context_pair_counter = 0
    dialogue_counter = 1
    delimiter_context_dialogue = " CC "
    delimiter = " # "
    delimiter_start = " SS "
    delimiter_sep = " SEP "

    your_persona_description = ""
    add_to_test_data = False
    add_to_valid_data = False

    arr_len_description = []
    arr_len_question = []
    arr_len_answer = []

    append_description = False

    with open(filename) as fp:
        question_line = ""
        for line in fp:
            data = []
            if line == '\n':
                question_line = ""
                dialogue_counter += 1
                if dialogue_counter % 5 == 0:
                    add_to_valid_data = True
                else:
                    add_to_valid_data = False
                    if random.randint(0, 100) < 5:
                        add_to_test_data = True
                        test_data.append("\n")
                        test_data.append("\n")
                    else:
                        add_to_test_data = False
            your_persona = re.findall(r"(your persona:.*\\n)", line)
            if with_description and len(your_persona) > 0:
                append_description = True
                your_persona = re.sub(r"\\n", '', your_persona[0]).split("your persona: ")
                your_persona_description = delimiter.join(your_persona[1:])
                your_persona_description = JOIN_TOKEN.join(tokenize(your_persona_description, nlp)[1])
                arr_len_description.append(len(your_persona_description.split()))
                #   persona # persona # persona # persona <context delimiter>
                question_line += your_persona_description + delimiter_context_dialogue
            line = re.sub(r"(your persona:.*\\n)", ' ', line)
            line = ' '.join(line.split())
            question = re.findall(r"text:(.*)labels:", line)
            answer = re.findall(r"labels:(.*)episode_done:", line)
            if len(answer) == 0:
                answer = re.findall(r"labels:(.*)question:", line)
            if len(answer) and len(question):
                question = JOIN_TOKEN.join(tokenize(question[0], nlp)[1])
                answer = JOIN_TOKEN.join(tokenize(answer[0], nlp)[1])
                question = question.replace('_ _ SILENCE _ _', '#S#')
                answer = answer.replace('_ _ SILENCE _ _', '#S#')
                arr_len_question.append(len(question.split()))
                arr_len_answer.append(len(answer.split()))

                if append_description:
                    append_description = False
                    data.append(question_line)
                    data.append(question)
                    question_line += delimiter_start
                if context_pair_counter < context_pair_count or context_pair_count == 0:
                    question_line += question + delimiter
                    context_pair_counter += 1
                else:
                    question_line = your_persona_description + delimiter_context_dialogue + delimiter_sep + question
                    context_pair_counter = 0

                data.append(question_line)
                data.append(answer)
                if add_to_valid_data:
                    valid_data += data
                elif add_to_test_data:
                    test_data.append(question)
                    test_data.append(answer)
                else:
                    train_data += data
                question_line = question_line + answer + delimiter
    return train_data, valid_data, test_data


def split_sentences_both_Persona_chat(filename):
    with open(filename) as fp:
        sentences = []
        for line in fp:
            is_description, y, p = personas_description(line)
            if is_description:
                if y != "":
                    sentences.append(y)
                if p != "":
                    sentences.append(p)
            sentences_splitted = line.split("\t")
            if len(sentences_splitted) > 1:
                utterance1 = re.findall(r"\d+ (.*)", sentences_splitted[0])[0]
                utterance2 = sentences_splitted[1]
                sentences.append(utterance1)
                sentences.append(utterance2)
    return sentences


# Collecting data for creating IDF for BART
def split_sentences_both_Persona_chat(filename):
    with open(filename) as fp:
        sentences = []
        for line in fp:
            is_description, y, p = personas_description(line)
            if is_description:
                if y != "":
                    sentences.append(y)
                if p != "":
                    sentences.append(p)
            sentences_splitted = line.split("\t")
            if len(sentences_splitted) > 1:
                utterance1 = re.findall(r"\d+ (.*)", sentences_splitted[0])[0]
                utterance2 = sentences_splitted[1]
                sentences.append(utterance1)
                sentences.append(utterance2)
    return sentences


def prepare_joke_dataset(nlp, reddit_jokes, stupidstuff, wocka):
    data_reddit = load_json(reddit_jokes)
    data_stupidstuff = load_json(stupidstuff)
    data_wocka = load_json(wocka)
    jokes_train = []
    jokes_valid = []
    all_data = [data_reddit, data_stupidstuff, data_wocka]
    counter = 0
    for i in range(len(all_data)):
        # i in [0:2]
        for joke in all_data[i]:
            counter += 1
            if i == 0:
                joint_joke = joke["title"] + " " + joke["body"]
            else:
                joint_joke = joke["body"]
            joint_joke = JOIN_TOKEN.join(tokenize(joint_joke, nlp)[1])
            if len(joint_joke.split()) > 1000:
                continue
            if counter % 5 == 0:
                jokes_valid.append(joint_joke)
            else:
                jokes_train.append(joint_joke)
    return jokes_train, jokes_valid


def prepare_lm_dataset(nlp, filename):
    train_data = []
    valid_data = []
    counter = 0
    with open(filename, "r") as file:
        for line in file:
            counter += 1
            if counter % 5 == 0:
                valid_data.append(JOIN_TOKEN.join(tokenize(line, nlp)[1]))
            else:
                train_data.append(JOIN_TOKEN.join(tokenize(line, nlp)[1]))
    return train_data, valid_data


def prepare_short_jokes(nlp, jokes_file):
    print("[Creating jokes dictionary]")
    words_dict = dict()
    jokes = load_csv(jokes_file)
    stop_words = set(stopwords.words('english'))
    filtered_jokes = []
    total_words = 0
    for i in range(len(jokes)):
        if i % 2 == 0:
            continue
        joke = JOIN_TOKEN.join(tokenize(jokes[i], nlp)[1])
        # deleting stop words
        filtered_joke = [j for j in joke.split() if not j.lower() in stop_words]
        for word in filtered_joke:
            total_words += 1
            word = word.lower()
            if word in words_dict:
                words_dict[word] += 1
            else:
                words_dict[word] = 1
        filtered_jokes.append(JOIN_TOKEN.join(filtered_joke))
    for w, c in words_dict.items():
        words_dict[w] = c / total_words
    return words_dict


def tokenize(text: string, t):
    tokens = [tok for tok in t.tokenizer(text) if not tok.text.isspace()]
    text_tokens = [tok.text for tok in tokens]
    return tokens, text_tokens


def create_custom_tokenizer(nlp):
    print("[Creating custom tokenizer]")
    custom_prefixes = [r'[0-9]+', r'\~', r'\–', r'\—', r'\$']
    custom_infixes = [r'[!&:,()]', r'\.', r'\-', r'\–', r'\—', r'\$']
    custom_suffixes = [r'\.', r'\–', r'\—', r'\$']
    default_prefixes = list(nlp.Defaults.prefixes) + custom_prefixes
    default_prefixes.remove(r'US\$')
    default_prefixes.remove(r'C\$')
    default_prefixes.remove(r'A\$')

    all_prefixes_re = spacy.util.compile_prefix_regex(tuple(default_prefixes))
    infix_re = spacy.util.compile_infix_regex(tuple(list(nlp.Defaults.infixes) + custom_infixes))
    suffix_re = spacy.util.compile_suffix_regex(tuple(list(nlp.Defaults.suffixes) + custom_suffixes))

    rules = dict(nlp.Defaults.tokenizer_exceptions)
    # remove "a." to "z." rules so "a." gets tokenized as a|.
    for c in range(ord("a"), ord("z") + 1):
        if f"{chr(c)}." in rules:
            rules.pop(f"{chr(c)}.")

    return Tokenizer(nlp.vocab, rules,
                     prefix_search=all_prefixes_re.search,
                     infix_finditer=infix_re.finditer, suffix_search=suffix_re.search,
                     token_match=None)


def prepare_dict():
    nlp = en_core_web_sm.load()
    nlp.tokenizer = create_custom_tokenizer(nlp)
    jokes_dict = prepare_short_jokes(nlp, DATASETS_PATH + 'shortjokes.csv')
    create_json(DATASETS_PATH + "jokes_dict.json", jokes_dict)


def prepare_decoding_feature_modifications():
    prepare_dict()
    sentences = split_sentences_both_Persona_chat(DATASETS_PATH + 'persona_chat_both.txt')
    with open(SAVE_DATA_PATH + 'idf', "wb") as fp:
        pickle.dump(sentences, fp)
    fp.close()


def prepare_lm_data(config):
    nlp = en_core_web_sm.load()
    nlp.tokenizer = create_custom_tokenizer(nlp)
    if not os.path.exists(SAVE_DATA_PATH[:-1]):
        os.makedirs(SAVE_DATA_PATH[:-1])

    print("[Preparing Joke data]")
    filename_train_jokes = SAVE_DATA_PATH + 'jokes_'
    if (config["model_lm_type"] == 'GPT2' and not os.path.isfile(filename_train_jokes + "train_gpt2")) or (
            config["model_lm_type"] == 'LSTM' and not os.path.isfile(filename_train_jokes + "train.csv")):
        train_data_jokes, valid_data_jokes = prepare_joke_dataset(nlp, DATASETS_PATH + 'reddit_jokes.json',
                                                                  DATASETS_PATH + 'stupidstuff.json',
                                                                  DATASETS_PATH + 'wocka.json')
        if config["model_lm_type"] == 'GPT2':
            save_data_for_GPT2(filename_train_jokes + "train_gpt2", train_data_jokes)
        elif config["model_lm_type"] == "LSTM":
            save_csv_row(filename_train_jokes + "train.csv", train_data_jokes)
            save_csv_row(filename_train_jokes + "valid.csv", valid_data_jokes)

    print("[Preparing Poetic data]")
    filename_train_poetic = SAVE_DATA_PATH + 'poetic_'
    if (config["model_lm_type"] == 'GPT2' and not os.path.isfile(filename_train_poetic + "train_gpt2")) or (
            config["model_lm_type"] == 'LSTM' and not os.path.isfile(filename_train_poetic + "train.csv")):
        train_data_poetic, valid_data_poetic = prepare_lm_dataset(nlp, DATASETS_PATH + "shakespeare.txt")
        # train_data_poetic = prepare_poetic_data(DATASETS_PATH + 'kaggle_poem_dataset.csv', "Content")
        if config["model_lm_type"] == 'GPT2':
            save_data_for_GPT2(filename_train_poetic + "train_gpt2", train_data_poetic)
        elif config["model_lm_type"] == "LSTM":
            save_csv_row(filename_train_poetic + "train.csv", train_data_poetic)
            save_csv_row(filename_train_poetic + "valid.csv", valid_data_poetic)

    print("[Preparing SST dataset]")
    filename_train_negative = SAVE_DATA_PATH + 'negative_'
    if (config["model_lm_type"] == 'GPT2' and not os.path.isfile(filename_train_negative + "train_gpt2")) or (
            config["model_lm_type"] == 'LSTM' and not os.path.isfile(filename_train_negative + "train.csv")):
        train_data_negative, valid_data_negative = prepare_lm_dataset(nlp, DATASETS_PATH + "sst_negative_sentences.txt")
        if config["model_lm_type"] == 'GPT2':
            save_data_for_GPT2(filename_train_negative + "train_gpt2", train_data_negative)
        elif config["model_lm_type"] == "LSTM":
            save_csv_row(filename_train_negative + "train.csv", train_data_negative)
            save_csv_row(filename_train_negative + "valid.csv", valid_data_negative)

    filename_train_positive = SAVE_DATA_PATH + 'positive_'
    if (config["model_lm_type"] == 'GPT2' and not os.path.isfile(filename_train_positive + "train_gpt2")) or (
            config["model_lm_type"] == 'LSTM' and not os.path.isfile(filename_train_positive + "train.csv")):
        train_data_positive, valid_data_positive = prepare_lm_dataset(nlp, DATASETS_PATH + 'sst_positive_sentences.txt')
        if config["model_lm_type"] == 'GPT2':
            save_data_for_GPT2(filename_train_positive + "train_gpt2", train_data_positive)
        elif config["model_lm_type"] == "LSTM":
            save_csv_row(filename_train_positive + "train.csv", train_data_positive)
            save_csv_row(filename_train_positive + "valid.csv", valid_data_positive)


def prepare_seq2seq_data(config):
    nlp = en_core_web_sm.load()
    nlp.tokenizer = create_custom_tokenizer(nlp)
    if not os.path.exists(SAVE_DATA_PATH[:-1]):
        os.makedirs(SAVE_DATA_PATH[:-1])

    if config["dataset_type_seq2seq"] == "PERSONA":
        print("[Preparing Persona data]")
        filename_train = SAVE_DATA_PATH + 'persona_train.csv'
        filename_valid = SAVE_DATA_PATH + 'persona_valid.csv'
        filename_test = SAVE_DATA_PATH + 'persona_test.csv'
        train_data, valid_data, test_data = prepare_Persona_chat(nlp, DATASETS_PATH + 'persona_chat.txt',
                                                                 config["context_pair_count"],
                                                                 config["with_description"])
    elif config["dataset_type_seq2seq"] == "PERSONA_BOTH":
        print("[Preparing Persona data with both persona description]")
        filename_train = SAVE_DATA_PATH + 'persona_train.csv'
        filename_valid = SAVE_DATA_PATH + 'persona_valid.csv'
        filename_test = SAVE_DATA_PATH + 'persona_test.csv'
        train_data, valid_data, test_data = prepare_both_Persona_chat(nlp, DATASETS_PATH + 'persona_chat_both.txt')

    if config["pretraining_dataset"] == "TWITTER":
        print("[Preparing Twitter data]")
        filename_train_pretraining = SAVE_DATA_PATH + 'pre_train.csv'
        filename_valid_pretraining = SAVE_DATA_PATH + 'pre_valid.csv'
        filename_test_pretraining = SAVE_DATA_PATH + 'pre_test.csv'
        train_data_pretraining, valid_data_pretraining, test_data_pretraining = prepare_Twitter_data(nlp,
                                                                                                     DATASETS_PATH + 'twitter_chat.txt')

    print("[train data: ", int(len(train_data) / 2), "]")
    print("[valid data: ", int(len(valid_data) / 2), "]")
    print("[test data:  ", int(len(test_data) / 2), "]")

    if config["model_seq2seq_type"] == 'BART':
        print("[Preparing data for BART]")
        save_data_for_BART(SAVE_DATA_PATH + "train", train_data)
        save_data_for_BART(SAVE_DATA_PATH + "val", valid_data)
        save_data_for_BART(SAVE_DATA_PATH + "test", test_data)

        save_data_for_BART(SAVE_DATA_PATH + "pre_train", train_data_pretraining)
        save_data_for_BART(SAVE_DATA_PATH + "pre_valid", valid_data_pretraining)
        save_data_for_BART(SAVE_DATA_PATH + "pre_test", test_data_pretraining)

    elif config["model_seq2seq_type"] == 'Basemodel':
        save_to_csv(filename_train, train_data)
        save_to_csv(filename_valid, valid_data)
        save_to_csv(filename_test, test_data)

        save_to_csv(filename_train_pretraining, train_data_pretraining)
        save_to_csv(filename_valid_pretraining, valid_data_pretraining)
        save_to_csv(filename_test_pretraining, test_data_pretraining)
