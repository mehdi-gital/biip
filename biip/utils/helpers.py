import os
import logging
import torch


def get_logger(log_path, prefix):
    logging.basicConfig(
        filename=os.path.join(log_path, f"{prefix}.log"),
        filemode="w",
        format="%(levelname)s %(asctime)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    return logger


def get_device(device):
    if device == "cpu":
        device = torch.device("cpu")
    elif device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise ValueError(
            "[!] Device is set to cuda in configs.json but cuda is not available. "
            "Please set the device to be cpu."
        )
    return device
