def save_data_for_BART(filename, data):
    """

    Args:
        filename: save file direction, data saved to the filename with format .source, .target
        data: data for saving to the file, data contains line by line source, target.

    Returns: None

    """
    with open(filename + ".source", 'w') as source, open(filename + ".target", 'w') as target:
        for i in range(0, len(data)-1, 2):
            source.write(data[i] + '\n')
            target.write(data[i + 1] + '\n')

    source.close()
    target.close()


def save_data_for_GPT2(filename, data):
    """

    Args:
        filename: save file direction
        data: data for saving to the file

    Returns: None

    """
    with open(filename, "w") as f:
        for i in range(len(data)):
            f.write(data[i] + "\n")
            f.write("\n")

    f.close()
