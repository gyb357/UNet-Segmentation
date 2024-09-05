from torch import Tensor
import torch
import numpy as np


def miou_coefficient(outputs: Tensor, masks: Tensor, smooth: float = 1e-10) -> Tensor:
    output = (torch.sigmoid(outputs) > 0.5).float() # output = torch.softmax(outputs)
    output = output.contiguous().view(-1)
    mask = masks.contiguous().view(-1)

    intersect = torch.logical_and(output, mask).sum().float().item()
    union = torch.logical_or(output, mask).sum().float().item()

    iou = (intersect + smooth)/(union + smooth)
    return np.nanmean(iou)


def miou_loss(outputs: Tensor, masks: Tensor, smooth: float = 1e-10) -> Tensor:
    return 1 - miou_coefficient(outputs, masks, smooth)

