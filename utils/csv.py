import csv
from collections import defaultdict
from itertools import zip_longest


def save_to_csv(name, lines):
    # counter_iam = 0
    with open(name, mode='w') as csv_file:
        fieldnames = ['question', 'answer']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        if len(lines) % 2 != 0:
            lines = lines[:-1]
        for i in range(0, len(lines), 2):
            # if str.lower(lines[i+1].split()[0]) == "i" and str.lower(lines[i+1].split()[1]) == "am":
            #     counter_iam += 1
            writer.writerow({'question': lines[i], 'answer': lines[i + 1]})
    # print("COUNTER I AM: ", counter_iam)


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


def load_histogram_data(filename):
    columns = defaultdict(list)
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k, v) in row.items():
                if v != "":
                    columns[k].append(int(v))
    return columns
