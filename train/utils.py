import yaml
import numpy as np
from typing import Dict, Any
from torch import Tensor


@staticmethod
def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        path (str): Path to the configuration file.
    """

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

@staticmethod
def tensor2numpy(tensor: Tensor) -> np.float32:
    """
    Convert a PyTorch tensor to a NumPy array.
    
    Args:
        tensor (Tensor): Input tensor.
    """

    return tensor.detach().cpu().numpy().astype(np.float32)

