import csv
from itertools import zip_longest


def save_data_in_column(filename, data_to_zip, columns_names):
    """

    Args:
        filename: save file direction
        data_to_zip: data what should be saved to columns
        columns_names: columns names

    Returns: None

    """
    with open(filename, 'w', newline='') as csv_file:
        export_data = zip_longest(data_to_zip, fillvalue='')
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        wr.writerow(columns_names)
        wr.writerows(export_data)
    csv_file.close()


def process_sst_to_csv():
    lengths = []
    with open('../datasets/sst_positive_sentences.txt', 'r') as file:
        for line in file:
            print(line)
            if len(line.split()) != 0:
                lengths.append(len(line.split()))

    save_data_in_column("sst_pos_lengths.csv", lengths, ["lengths"])


process_sst_to_csv()
