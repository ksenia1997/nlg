import matplotlib.pyplot as plt


def plot_histogram(title, xlabel, ylabel):
    x = [0, 1, 2]
    plt.hist(x, bins=3)
    plt.xlabel(xlabel)
    plt.ylabel()
    plt.title(title)
    plt.show()
