import os
import random
import re
import string

import en_core_web_sm
import spacy
from nltk.corpus import stopwords
from spacy.tokenizer import Tokenizer

from params import *
from utils.csv_process import *
from utils.json_process import *


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
    print("Prepare Persona chat with both descriptions")
    with open(filename) as fp:
        your_persona_description = []
        partner_persona_description = []
        counter = 0
        delimiter_context_dialogue = " CC "
        delimiter = " # "
        delimiter_start = " SS "
        delimiter_sep = " SEP "
        delimiter_end = " EE "

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
    print("Reading Twitter data")
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
    print("Reading Persona chat")
    train_data = []
    test_data = []
    valid_data = []
    context_pair_counter = 0
    dialogue_counter = 1
    delimiter_context_dialogue = " CC "
    delimiter = " # "
    delimiter_start = " SS "
    delimiter_sep = " SEP "
    delimiter_end = " EE "

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


def prepare_joke_dataset(nlp, reddit_jokes, stupidstuff, wocka):
    data_reddit = load_json(reddit_jokes)
    data_stupidstuff = load_json(stupidstuff)
    data_wocka = load_json(wocka)
    jokes_train, jokes_valid, jokes_test = [], [], []
    counter = 0
    all_data = [data_reddit, data_stupidstuff, data_wocka]
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
            if random.randint(0, 100) < 5:
                jokes_test.append(joint_joke)
            elif counter % 5 == 0:
                jokes_valid.append(joint_joke)
            else:
                jokes_train.append(joint_joke)
    return jokes_train, jokes_valid, jokes_test


def prepare_short_jokes(nlp, jokes_file):
    words_dict = dict()
    jokes = load_csv(jokes_file)
    stop_words = set(stopwords.words('english'))
    filtered_jokes = []
    for i in range(len(jokes)):
        if i % 2 == 0:
            continue
        joke = JOIN_TOKEN.join(tokenize(jokes[i], nlp)[1])
        # deleting stop words
        filtered_joke = [j for j in joke.split() if not j.lower() in stop_words]
        for word in filtered_joke:
            word = word.lower()
            if word in words_dict:
                words_dict[word] += 1
            else:
                words_dict[word] = 1
        filtered_jokes.append(JOIN_TOKEN.join(filtered_joke))
    total_words = len(words_dict)
    for w, c in words_dict.items():
        words_dict[w] = c / total_words
    return words_dict


def tokenize(text: string, t):
    tokens = [tok for tok in t.tokenizer(text) if not tok.text.isspace()]
    text_tokens = [tok.text for tok in tokens]
    return tokens, text_tokens


def create_custom_tokenizer(nlp):
    print("Creating custom tokenizer")
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


def prepare_dict(config):
    nlp = en_core_web_sm.load()
    nlp.tokenizer = create_custom_tokenizer(nlp)
    jokes_dict = prepare_short_jokes(nlp, DATASETS_PATH + 'shortjokes.csv')
    create_json(DATASETS_PATH + "jokes_dict.json", jokes_dict)


def prepare_data(config):
    print("Prepare data")
    nlp = en_core_web_sm.load()
    nlp.tokenizer = create_custom_tokenizer(nlp)
    if not os.path.exists(SAVE_DATA_PATH[:-1]):
        os.makedirs(SAVE_DATA_PATH[:-1])

    if config["data_type"] == "PERSONA":
        filename_train = SAVE_DATA_PATH + 'persona_train.csv'
        filename_valid = SAVE_DATA_PATH + 'persona_valid.csv'
        filename_test = SAVE_DATA_PATH + 'persona_test.csv'
        train_data, valid_data, test_data = prepare_Persona_chat(nlp, DATASETS_PATH + 'persona_chat.txt',
                                                                 config["context_pair_count"],
                                                                 config["with_description"])

    elif config["data_type"] == "TWITTER":
        filename_train = SAVE_DATA_PATH + 'twitter_train.csv'
        filename_valid = SAVE_DATA_PATH + 'twitter_valid.csv'
        filename_test = SAVE_DATA_PATH + 'twitter_test.csv'
        train_data, valid_data, test_data = prepare_Twitter_data(nlp, DATASETS_PATH + 'twitter_chat.txt')

    elif config["data_type"] == "PERSONA_BOTH":
        filename_train = SAVE_DATA_PATH + 'train.csv'
        filename_valid = SAVE_DATA_PATH + 'valid.csv'
        filename_test = SAVE_DATA_PATH + 'test.csv'
        train_data, valid_data, test_data = prepare_both_Persona_chat(nlp, DATASETS_PATH + 'persona_chat_both.txt')

    elif config["data_type"] == "JOKE":
        filename_train = SAVE_DATA_PATH + 'jokes_train.csv'
        filename_valid = SAVE_DATA_PATH + 'jokes_valid.csv'
        filename_test = SAVE_DATA_PATH + 'jokes_test.csv'
        train_data, valid_data, test_data = prepare_joke_dataset(nlp, DATASETS_PATH + 'reddit_jokes.json',
                                                                 DATASETS_PATH + 'stupidstuff.json',
                                                                 DATASETS_PATH + 'wocka.json')

    print("train data: ", len(train_data) / 2)
    print("valid data: ", len(valid_data) / 2)
    print("test data: ", len(test_data) / 2)

    if config["data_BART"]:
        process_data_for_BART(SAVE_DATA_PATH + "train", train_data)
        process_data_for_BART(SAVE_DATA_PATH + "val", valid_data)
        process_data_for_BART(SAVE_DATA_PATH + "test", test_data)
        return

    if config["data_type"] in ["TWITTER", "PERSONA_BOTH", "PERSONA"]:
        save_to_csv(filename_train, train_data)
        save_to_csv(filename_valid, valid_data)
        save_to_csv(filename_test, test_data)
    elif config["data_type"] in ["JOKE"]:
        save_csv_row(filename_train, train_data)
        save_csv_row(filename_valid, valid_data)
        save_csv_row(filename_test, test_data)
