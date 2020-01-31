import random
import re
import string

import en_core_web_sm
import spacy
from spacy.tokenizer import Tokenizer

from params import *
from utils.csv import *


def prepare_Twitter_data(filename):
    print("Reading Twitter data")
    train_data = []
    valid_data = []
    test_data = []
    counter = 0
    with open(filename) as fp:
        for line in fp:
            train_data.append(line[:15])
            if counter % 10 == 0:
                valid_data.append(line)
            if counter % 20 == 0:
                test_data.append(line)
            counter += 1

    return train_data, valid_data, test_data


def prepare_Persona_chat(filename, context_pair_count):
    print("Reading Persona chat")
    train_data = []
    test_data = []
    valid_data = []
    context_pair_counter = 0
    dialogue_counter = 1

    your_persona_description = ""
    add_to_test_data = False
    add_to_valid_data = False
    with open(filename) as fp:
        question_line = ""
        for line in fp:
            if line == '\n':
                question_line = ""
                dialogue_counter += 1
                if random.randint(0, 100) < 5:
                    add_to_test_data = True
                    test_data.append("\n")
                    test_data.append("\n")
                else:
                    add_to_test_data = False
                if dialogue_counter % 5 == 0:
                    add_to_valid_data = True
                else:
                    add_to_valid_data = False
            your_persona = re.findall(r"(your persona:.*\\n)", line)
            if WITH_DESCRIPTION and len(your_persona) > 0:
                your_persona = re.sub(r"\\n", '', your_persona[0]).split("your persona: ")
                your_persona_description = ' # '.join(your_persona[1:])
                print("your persona description: ", your_persona_description)
                question_line += your_persona_description
            line = re.sub(r"(your persona:.*\\n)", ' ', line)
            line = ' '.join(line.split())
            print("Line without description: ", line)
            question = re.findall(r"text:(.*)labels:", line)
            answer = re.findall(r"labels:(.*)episode_done:", line)
            print("Question: ", question)
            print("Answer: ", answer)
            if len(answer) == 0:
                answer = re.findall(r"labels:(.*)question:", line)
            print("Also answer: ", answer)
            if len(answer) and len(question):
                if add_to_valid_data:
                    print("VALID")
                    question_line += " # " + question[0]
                    answer_line = question_line + " # " + answer[0]
                    valid_data.append(question_line)
                    valid_data.append(answer_line)
                    question_line = answer_line
                elif add_to_test_data:
                    print("TEST")
                    test_data.append(question[0])
                    test_data.append(answer[0])
                else:
                    print("TRAIN")
                    if context_pair_counter < context_pair_count or context_pair_count == 0:
                        question_line += " # " + question[0]
                        context_pair_counter += 1
                    else:
                        question_line = your_persona_description + " # " + question[0]
                        context_pair_counter = 0
                    answer_line = question_line + " # " + answer[0]
                    print("QUESTION: ", question_line)
                    print("ANSWER: ", answer_line)
                    train_data.append(question_line)
                    train_data.append(answer_line)
                    question_line = answer_line

    return train_data, valid_data, test_data


def tokenize(text: string, t):
    tokens = [tok for tok in t.tokenizer(text) if not tok.text.isspace()]
    text_tokens = [tok.text for tok in tokens]
    return tokens, text_tokens


def tokenize_and_join(text, t, jointoken=JOIN_TOKEN):
    tokenized_text = []
    for sentnence in text:
        tokenized_text.append(jointoken.join(tokenize(sentnence, t)[1]))
    return tokenized_text


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


nlp = en_core_web_sm.load()
nlp.tokenizer = create_custom_tokenizer(nlp)


def prepare_data():
    print("Prepare data")
    if DATA_TYPE == "PERSONA":
        train_data, valid_data, test_data = prepare_Persona_chat('persona_chat.txt', CONTEXT_PAIR_COUNT)
    elif DATA_TYPE == "TWITTER":
        train_data, valid_data, test_data = prepare_Twitter_data('twitter_chat.txt')

    tokenized_train_data = tokenize_and_join(train_data, nlp)
    tokenized_valid_data = tokenize_and_join(valid_data, nlp)
    tokenized_test_data = tokenize_and_join(test_data, nlp)

    print("train data: ", len(tokenized_train_data))
    print("valid data: ", len(tokenized_valid_data))
    print("test data: ", len(tokenized_test_data))

    save_to_csv('train.csv', tokenized_train_data)
    save_to_csv('valid.csv', tokenized_valid_data)
    save_to_csv('test.csv', tokenized_test_data)
