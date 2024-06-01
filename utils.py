from torch import Tensor
import numpy as np


def tensor_to_numpy(tensor: Tensor) -> float:
    return tensor.detach().cpu().numpy().astype(np.float32)

