from torch import Tensor
import torch


def iou_coef(inputs: Tensor, masks: Tensor, epsilon: float = 1e-6) -> float:
    inputs = (torch.sigmoid(inputs) > 0.5).float()

    # [Batch x 1 x H x W] -> [Batch x H x W]
    inputs = inputs.squeeze(1)
    masks = masks.squeeze(1)

    assert inputs.shape == masks.shape
    assert masks.min() >= 0 and masks.max() <= 1
    
    sum_dim = (-1, -2)
    inter = (inputs*masks).sum(dim=sum_dim)
    union = (inputs + masks - inputs*masks).sum(dim=sum_dim)
    iou = (inter + epsilon)/(union + epsilon)
    return iou.mean()

