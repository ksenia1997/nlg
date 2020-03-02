import csv
from itertools import zip_longest


def save_to_csv(name, lines):
    with open(name, mode='w') as csv_file:
        fieldnames = ['source', 'target']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        if len(lines) % 2 != 0:
            lines = lines[:-1]
        for i in range(0, len(lines), 2):
            writer.writerow({'source': lines[i], 'target': lines[i + 1]})


def save_csv_row(name, lines):
    with open(name, mode='w') as csv_file:
        fieldnames = ['source']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(lines)):
            writer.writerow({'source': lines[i]})


def load_csv(name):
    with open(name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        lines = []
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                lines.append(row[0])
                lines.append(row[1])
                line_count += 1
    return lines


def save_data_in_column(filename, data_to_zip, columns_name):
    with open(filename, 'w', newline='') as myfile:
        export_data = zip_longest(*data_to_zip, fillvalue='')
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(columns_name)
        wr.writerows(export_data)
    myfile.close()
