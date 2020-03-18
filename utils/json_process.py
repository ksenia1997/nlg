import json


def create_json(filename, data):
    """

    Args:
        filename: save file direction
        data: data for saving to the file

    Returns: None

    """
    with open(filename, 'w') as f:
        json.dump(data, f)
    f.close()


def load_json(filename):
    """

    Args:
        filename: load file direction

    Returns: loaded data

    """

    with open(filename, 'r') as f:
        return json.load(f)


def process_data_to_json(filename, data):
    """

    Args:
        filename: save file direction
        data: data for saving to the file, data contains line by line source, target.

    Returns: None

    """
    json_data = []
    length_data = len(data)
    if len(data) % 2 != 0:
        length_data = len(data) - 1
    for i in range(0, length_data, 2):
        json_data.append({'src': data[i], 'trg': data[i + 1]})

    create_json(filename, json_data)



