import yaml
import numpy as np
from typing import Any, Dict, List
from pathlib import Path
from torch import Tensor
from model.unet import UNet
from model.unet2plus import UNet2Plus
from model.unet3plus import UNet3Plus


@staticmethod
def ternary_operation(a: bool, b: Any, c: Any) -> Any:
    """
    Args:
        a (bool): Condition to check
        b (Any): Value to return if condition is `True`
        c (Any): Value to return if condition is `False`
    """

    return b if a else c

@staticmethod
def ternary_operation_elif(a: bool, b: Any, c: bool, d: Any, e: Any) -> Any:
    """
    Args:
        a (bool): Condition to check
        b (Any): Value to return if condition is `True`
        c (bool): Condition to check if `a` is `False`
        d (Any): Value to return if `c` is `True`
        e (Any): Value to return if `c` is `False`
    """

    return b if a else ternary_operation(c, d, e)

@staticmethod
def load_config(path: str) -> Dict[str, Any]:
    """
    Args:
        path (str): Path to the configuration file
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
    Args:
        tensor (Tensor): Input tensor
    """

    return tensor.detach().cpu().numpy().astype(np.float32)

@staticmethod
def get_model_list(model_ensemble: List[str]) -> List[str]:
    """
    Args:
        model_ensemble (List[str]): List of model names
    """

    model_list = []
    for model_name in model_ensemble:
        if model_name == 'unet':
            model_list.append(UNet)
        elif model_name == 'unet2plus':
            model_list.append(UNet2Plus)
        elif model_name == 'unet3plus':
            model_list.append(UNet3Plus)
    return model_list

