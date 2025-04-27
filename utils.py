import yaml
import numpy as np
import torch.nn as nn
import os
from typing import Any, Dict
from pathlib import Path
from torch import Tensor


def ternary_operation(a: bool, b: Any, c: Any) -> Any:
    """
    Args:
        a (bool): Condition to check
        b (Any): Value to return if condition is `True`
        c (Any): Value to return if condition is `False`
    """

    return b if a else c

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

def tensor2numpy(tensor: Tensor) -> np.float32:
    """
    Args:
        tensor (Tensor): Input tensor
    """

    return tensor.detach().cpu().numpy().astype(np.float32)

def get_model(model_name: str) -> nn.Module:
    """
    Args:
        model_name (str): Name of the model
    """

    # Import models
    from model.unet import UNet, UNet2Plus, UNet3Plus

    if model_name == 'unet':
        return UNet
    elif model_name == 'unet2plus':
        return UNet2Plus
    elif model_name == 'unet3plus':
        return UNet3Plus
    else:
        raise ValueError(f"Model {model_name} not recognized. Please check the model name in the config file.")

def has_files(directory: str) -> bool:
    """
    Args:
        directory (str): Directory to check
    """

    try:
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name != '.gitkeep':
                return True
        return False
    except FileNotFoundError:
        raise ValueError(f"Directory '{directory}' does not exist.")

