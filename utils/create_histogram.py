import matplotlib.pyplot as plt


def plot_histogram(title, xlabel, ylabel, data, bins_number, filename):
    plt.hist(data, bins=bins_number, color='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    # plt.show()
