import csv
from itertools import zip_longest


def save_to_csv(filename, lines):
    """

    Args:
        filename: save file direction
        lines: data for saving, first line must be source, second line must be target etc.

    Returns: None

    """
    with open(filename, mode='w') as csv_file:
        fieldnames = ['source', 'target']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        if len(lines) % 2 != 0:
            lines = lines[:-1]
        for i in range(0, len(lines), 2):
            writer.writerow({'source': lines[i], 'target': lines[i + 1]})
    csv_file.close()


def save_csv_row(name, lines):
    with open(name, mode='w') as csv_file:
        fieldnames = ['source']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(lines)):
            writer.writerow({'source': lines[i]})


def load_csv(filename):
    """

    Args:
        filename: load file direction

    Returns: loaded data

    """
    lines = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                lines.append(row[0])
                lines.append(row[1])
                line_count += 1
    csv_file.close()
    return lines


def save_data_in_column(filename, data_to_zip, columns_names):
    """

    Args:
        filename: save file direction
        data_to_zip: data what should be saved to columns
        columns_names: columns names

    Returns: None

    """
    with open(filename, 'w', newline='') as csv_file:
        export_data = zip_longest(*data_to_zip, fillvalue='')
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        wr.writerow(columns_names)
        wr.writerows(export_data)
    csv_file.close()


def prepare_poetic_data(file_csv, column_name):
    first_line = True
    data = []
    with open(file_csv, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if first_line:
                first_line = False
                idx_column_name = row.index(column_name)
                continue
            data.append(row[idx_column_name] + "\n")
    csv_file.close()
    return data
