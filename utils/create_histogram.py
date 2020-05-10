import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def plot_histogram(title, xlabel, ylabel, data1, data2, label1, label2, filename):
    if label1 is None and label2 is None and data2 is None:
        plt.hist(data1, bins=50)
    else:
        n, bins, patches = plt.hist([data1, data2], histtype='bar', label=[label1, label2])
        plt.legend(prop={'size': 10})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()
    # plt.show()


def load_histogram_data(filename):
    """

    Args:
        filename: load file direction

    Returns: loaded data from csv

    """
    columns = defaultdict(list)
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k, v) in row.items():
                if v != "":
                    columns[k].append(int(v))
    return columns


def create_persona_histograms():
    columns = load_histogram_data('../datasets/description.csv')
    yp_desc = columns['Your persona description length']
    pp_desc = columns['Partner\'s persona description length']
    utr1_length = columns['utterance1 length']
    utr2_length = columns['utterance2 length']
    yp_desc.sort()
    pp_desc.sort()
    utr1_length.sort()
    utr2_length.sort()
    plot_histogram('Histogram of persona description lengths', 'number of words in description',
                   'number of descriptions', yp_desc, pp_desc, 'person1', 'person2', 'persona_desc.pdf')
    plot_histogram('Histogram of sequences\' lengths', 'number of words in a sequence',
                   'number of sequences', utr1_length, utr2_length, 'person1', 'person2', 'uttr_length.pdf')


def create_joke_histogram():
    columns = load_histogram_data('../datasets/jokes_length.csv')
    jokes_length = columns['jokes_length']
    jokes_length.sort()
    plot_histogram('Histogram of jokes\' lengths', 'number of words in a joke',
                   'number of jokes', jokes_length, None, None, None, 'jokes_length.pdf')


def create_tweet_histogram():
    columns = load_histogram_data('../datasets/twitter_length.csv')
    tweet_lengths = columns['twit_length']
    tweet_lengths.sort()
    plot_histogram('Histogram of tweets\' lengths', 'number of words in a tweet',
                   'number of tweets', tweet_lengths, None, None, None, 'tweet.pdf')


def create_sst_histogram():
    neg_columns = load_histogram_data('../datasets/sst_negative_lengths.csv')
    neg_lengths = neg_columns['lengths']
    neg_lengths.sort()
    pos_columns = load_histogram_data('../datasets/sst_positive_lengths.csv')
    pos_lengths = pos_columns['lengths']
    pos_lengths.sort()
    plot_histogram('Histogram of SST sequences\' lengths', 'number of words in a sequence',
                   'number of sequences', neg_lengths, pos_lengths, 'negative', 'positive', 'sst.pdf')


def create_shakespear_histogram():
    columns = load_histogram_data('../datasets/shakespeare_lengths.csv')
    lengths = columns['lengths']
    lengths.sort()
    plot_histogram('Histogram of Shakespeare\'s plays', 'number of words in a sequence',
                   'number of sequences', lengths, None, None, None, 'shakespeare.pdf')


create_persona_histograms()
