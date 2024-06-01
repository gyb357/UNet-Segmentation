from torch import Tensor
import numpy as np
import yaml


def tensor_to_numpy(tensor: Tensor) -> float:
    return tensor.detach().cpu().numpy().astype(np.float32)


def load_config(config_file: str):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

