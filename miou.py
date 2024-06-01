from torch import Tensor


def miou_coef(inputs: Tensor, masks: Tensor, epsilon: float = 1e-6) -> Tensor:
    inputs = (inputs > 0.5).float()
    inputs = inputs.squeeze(1)
    masks = masks.squeeze(1)

    assert inputs.shape == masks.shape
    assert masks.min() >= 0 and masks.max() <= 1
    
    sum_dim = (-1, -2)
    inter = (inputs*masks).sum(dim=sum_dim)
    union = (inputs + masks - inputs*masks).sum(dim=sum_dim)
    iou = (inter + epsilon)/(union + epsilon)
    return iou.mean()


def miou_loss(inputs: Tensor, masks: Tensor, epsilon: float = 1e-6) -> Tensor:
    return 1 - miou_coef(inputs, masks, epsilon)

