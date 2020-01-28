import csv


def save_to_csv(name, lines):
    with open(name, mode='w') as csv_file:
        fieldnames = ['question', 'answer']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        if len(lines) % 2 != 0:
            lines = lines[:-1]
        for i in range(0, len(lines), 2):
            writer.writerow({'question': lines[i], 'answer': lines[i + 1]})


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
