import yaml
import numpy as np
from typing import Any, Dict
from pathlib import Path
from torch import Tensor


@staticmethod
def ternary_op(a: bool, b: Any, c: Any) -> Any:
    """
    Ternary operator implementation.

    Args:
        a (bool): Condition to check
        b (Any): Value to return if condition is `True`
        c (Any): Value to return if condition is `False`
    """

    return b if a else c

@staticmethod
def ternary_op_elif(a: bool, b: Any, c: bool, d: Any, e: Any) -> Any:
    """
    Ternary operator with `elif` implementation.
    
    Args:
        a (bool): Condition to check
        b (Any): Value to return if condition is `True`
        c (bool): Condition to check if `a` is `False`
        d (Any): Value to return if `c` is `True`
        e (Any): Value to return if `c` is `False`
    """
    
    return b if a else ternary_op(c, d, e)

@staticmethod
def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        path (str): Path to the configuration file.
    """

    # Convert path to Path object
    path = Path(path)

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

