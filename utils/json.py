import json


def create_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def process_data_to_json(filename, data):
    # data contains line by line source, target.
    json_data = []
    for i in range(0, len(data), 2):
        json_data.append({'src': data[i], 'trg': data[i + 1]})

    create_json(filename, json_data)


def process_data_for_BART(filename, data):
    with open(filename + ".source", 'w') as source, open(filename + ".target", 'w') as target:
        for i in range(0, len(data), 2):
            source.write(data[i])
            target.write(data[i])

    source.close()
    target.close()
