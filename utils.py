from torch import Tensor
import numpy as np
import yaml


def operate(a: bool, b: any, c: any) -> any:
    return b if a is True else c


def operate_elif(a: bool, b: any, c: bool, d: any, e: any) -> any:
    return b if a is True else operate(c, d, e)


def tensor_to_numpy(tensor: Tensor) -> np.float32:
    return tensor.detach().cpu().numpy().astype(np.float32)


def load_config(path: str) -> dict:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

