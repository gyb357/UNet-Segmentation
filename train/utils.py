import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any
from torch import Tensor
from typing import Optional


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


@staticmethod
def visualize(
    outputs: Tensor,
    masks: Tensor,
    num_classes: int,
    display_time: float = 3.0,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the predicted masks and ground truth masks.
    
    Args:
        outputs (Tensor): Predicted mask tensor
        masks (Tensor): Ground truth mask tensor
        num_classes (int): Number of classes
        display_time (float): Time to display each image (0 for no display)
        save_path (str): Path to save the visualization
    """

    # Check if display time is valid
    if display_time <= 0 and save_path is None:
        return
    
    if num_classes == 1:
        outputs = torch.sigmoid(outputs)
    else:
        outputs = torch.softmax(outputs, dim=1)

    
    plt.figure(figsize=(12, 6))
    