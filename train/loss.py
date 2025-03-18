import torch
from torch import Tensor


def miou_coefficient(
        outputs: Tensor,
        masks: Tensor,
        num_classes: int,
        smooth: float = 1e-6,
        threshold: float = 0.5
) -> float:
    """
    Compute the mean IoU (Intersection over Union) for a batch of predictions.

    Formula
    -------
       IoU = TP / (TP + FP + FN)
     - TP: True positive
     - FP: False positive
     - FN: False negative

    Args:
        outputs (Tensor): Predicted mask tensor
        masks (Tensor): Ground truth mask tensor
        num_classes (int): Number of classes
        smooth (float): Smoothing factor to avoid division by zero
        threshold (float): Threshold value for binarization
    """

    # Apply the activation function for binary or multi-class segmentation
    if num_classes == 1:
        outputs = torch.sigmoid(outputs)
    else:
        outputs = torch.softmax(outputs, dim=1)

    # Binarize outputs and masks for each class
    iou_list = []
    for i in range(num_classes):
        if num_classes > 1:
            output_class = (outputs[:, i] > threshold).float()
            mask_class = (masks[:, i] > threshold).float()
        else:
            output_class = (outputs > threshold).float()
            mask_class = (masks > threshold).float()

        # Flatten the tensors
        output_flat = output_class.view(-1)
        mask_flat = mask_class.view(-1)

        # Intersection and union
        intersection = torch.logical_and(output_flat, mask_flat).sum().float()
        union = torch.logical_or(output_flat, mask_flat).sum().float()
        iou = (intersection + smooth)/(union + smooth)

        iou_list.append(iou)
    return torch.mean(torch.tensor(iou_list)).item()


def dice_coefficient(
        outputs: Tensor,
        masks: Tensor,
        num_classes: int,
        smooth: float = 1e-6,
        threshold: float = 0.5
) -> float:
    """
    Compute the Dice coefficient for a batch of predictions.

    Formula
    -------
       Dice = (2 * TP) / (2 * TP + FP + FN)
     - TP: True positive
     - FP: False positive
     - FN: False negative

    Args:
        outputs (Tensor): Predicted mask tensor
        masks (Tensor): Ground truth mask tensor
        num_classes (int): Number of classes
        smooth (float): Smoothing factor to avoid division by zero
        threshold (float): Threshold value for binarization
    """

    # Apply the activation function for binary or multi-class segmentation
    if num_classes == 1:
        outputs = torch.sigmoid(outputs)
    else:
        outputs = torch.softmax(outputs, dim=1)

    # Binarize outputs and masks for each class
    dice_list = []
    for i in range(num_classes):
        if num_classes > 1:
            output_class = (outputs[:, i] > threshold).float()
            mask_class = (masks[:, i] > threshold).float()
        else:
            output_class = (outputs > threshold).float()
            mask_class = (masks > threshold).float()

        # Flatten the tensors
        output_flat = output_class.view(-1)
        mask_flat = mask_class.view(-1)

        # Intersection and dice
        intersection = torch.logical_and(output_flat, mask_flat).sum().float()
        sum_output_mask = output_flat.sum() + mask_flat.sum()
        dice = (2*intersection + smooth)/(sum_output_mask + smooth)

        dice_list.append(dice)
    return torch.mean(torch.tensor(dice_list)).item()


def miou_loss(
        outputs: Tensor,
        masks: Tensor,
        num_classes: int,
        smooth: float = 1e-6,
        threshold: float = 0.5
) -> Tensor:
    """
    Compute the mean IoU (Intersection over Union) loss for a batch of predictions.

    Args:
        outputs (Tensor): Predicted mask tensor
        masks (Tensor): Ground truth mask tensor
        num_classes (int): Number of classes
        smooth (float): Smoothing factor to avoid division by zero
        threshold (float): Threshold value for binarization
    """

    return 1.0 - miou_coefficient(outputs, masks, num_classes, smooth, threshold)


def dice_loss(
        outputs: Tensor,
        masks: Tensor,
        num_classes: int,
        smooth: float = 1e-6,
        threshold: float = 0.5
) -> Tensor:
    """
    Compute the Dice loss for a batch of predictions.

    Args:
        outputs (Tensor): Predicted mask tensor
        masks (Tensor): Ground truth mask tensor
        num_classes (int): Number of classes
        smooth (float): Smoothing factor to avoid division by zero
        threshold (float): Threshold value for binarization
    """

    return 1.0 - dice_coefficient(outputs, masks, num_classes, smooth, threshold)

