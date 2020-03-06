import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def plot_histogram(title, xlabel, ylabel, data, bins_number, filename):
    """

    Args:
        title: title of the histogram
        xlabel: title what is written on axis x
        ylabel: title what is written on axis y
        data: data for plotting
        bins_number: number of intervals
        filename: save file direction

    Returns: None

    """
    plt.hist(data, bins=bins_number, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
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
    plot_histogram('Histogram of your persona description lengths', 'number of words in description',
                   'number of descriptions', yp_desc, 50, 'persona_desc.pdf')
    plot_histogram('Histogram of partner\'s persona description lengths', 'number of words in description',
                   'number of descriptions', pp_desc, 50, 'partner_desc.pdf')
    plot_histogram('Histogram of utterances\' lengths of the first person', 'number of words in utterance',
                   'number of utterances', utr1_length, 50, 'uttr1_length.pdf')
    plot_histogram('Histogram of utterances\' lengths of the first person', 'number of words in utterance',
                   'number of utterances', utr2_length, 50, 'uttr2_length.pdf')


def create_joke_histogram():
    columns = load_histogram_data('../datasets/jokes_length.csv')
    jokes_length = columns['jokes_length']
    jokes_length.sort()
    plot_histogram('Histogram of jokes\' lengths', 'number of words in a joke',
                   'number of jokes', jokes_length, 50, 'jokes_length.pdf')


def create_tweet_histogram():
    columns = load_histogram_data('../datasets/twitter_length.csv')
    tweet_lengths = columns['twit_length']
    tweet_lengths.sort()
    plot_histogram('Histogram of tweets\' lengths', 'number of words in a tweet',
                   'number of tweets', tweet_lengths, 50, 'tweet.pdf')


create_tweet_histogram()
