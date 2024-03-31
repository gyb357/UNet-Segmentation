import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs*targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_bce = bce + dice_loss
        return Dice_bce


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs*targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        iou = (intersection + smooth)/(union + smooth)
        return 1 - iou


def dice_loss(
        inputs: Tensor,
        targets: Tensor,
        smooth: float = 1e-6
) -> float:
    sum_dim = (-1, -2)
    inputs = F.sigmoid(inputs)
    # Intersection between pred and target
    inter = (inputs*targets).sum(dim=sum_dim)
    union = inputs.sum(dim=sum_dim) + targets.sum(dim=sum_dim)
    # Avoid division by zero
    union = torch.where(union == 0, inter, union)
    # Dice coefficient
    dice = 1 - 2*(inter + smooth)/(union + smooth)
    return dice.mean()

