import os

import torch


def save_model_epoch(model, save_dir, model_name, epoch):
    """

    Args:
        model: nn model
        save_dir: save model direction
        model_name: model name
        epoch: number of epoch

    Returns: None

    """

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, model_name)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    print("save all model to {}".format(save_path))
    output = open(save_path, mode="wb")
    torch.save(model.state_dict(), output)
    output.close()


def save_best_model(model, save_path):
    """

    Args:
        model: nn model
        save_dir: save model direction
        best_model_name: model name

    Returns: None

    """
    print("save best model to {}".format(save_path))
    if os.path.exists(save_path):
        os.remove(save_path)
    output = open(save_path, mode="wb")
    torch.save(model.state_dict(), output)
    output.close()


def load_model(model, model_path, device):
    """

    Args:
        model: nn model
        model_path: load model direction
        device: device

    Returns: loaded model

    """
    try:
        if os.path.exists(model_path):
            return model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    except OSError:
        print("Cannot load model: ", model, "\n" + "Model path does not exist.")
