import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def plot_histogram(title, xlabel, ylabel, data, bins_number, filename):
    plt.hist(data, bins=bins_number, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    # plt.show()


def load_histogram_data(filename):
    columns = defaultdict(list)
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k, v) in row.items():
                if v != "":
                    columns[k].append(int(v))
    return columns


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
