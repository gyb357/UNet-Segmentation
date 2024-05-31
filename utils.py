import os
from torch import Tensor
import numpy as np


def makedirs(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def tensor_to_numpy(tensor: Tensor) -> np.float:
    return tensor.detach().cpu().numpy().astype(np.float32)

