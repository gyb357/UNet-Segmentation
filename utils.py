import json
from torch import Tensor
import numpy as np
import os


def load_config(path: str) -> json:
    with open(path, 'r') as f:
        return json.load(f)
    
def operate(a: bool, b, c):
    return b if a is True else c

def tensor_to_numpy(tensor: Tensor) -> np.float:
    return tensor.detach().cpu().numpy().astype(np.float32)

def make_folder(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

