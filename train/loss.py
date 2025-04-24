import torch
from torch import Tensor


class Loss():
    def __init__(
            self,
            num_classes: int,
            method: str,
            threshold: float = 0.5,
            epsilon: float = 1e-6,
    ) -> None:
        """
        Args:
            num_classes (int): The number of classes
            method (str): The method to compute the coefficient ('dice' or 'iou')
            threshold (float): The threshold value (default: 0.5)
            epsilon (float): The epsilon value (default: 1e-6)
        """

        # Attributes
        self.num_classes = num_classes
        self.method = method
        self.threshold = threshold
        self.epsilon = epsilon
        self.activation = torch.sigmoid if num_classes == 1 else lambda x: torch.softmax(x, dim=1)

    def get_coefficient(self, preds: Tensor, masks: Tensor) -> Tensor:
        """
        Args:
            preds (Tensor): The predicted outputs
            masks (Tensor): The ground truth masks
        """

        # Activation function
        preds = self.activation(preds)

        # Binarize
        coefs = []
        for c in range(self.num_classes):
            # Flatten per-class predictions and masks
            if self.num_classes > 1:
                pred_c = preds[:, c].view(-1)
                mask_c = masks[:, c].view(-1)
            else:
                pred_c = preds.view(-1)
                mask_c = masks.view(-1)

            # Binarize
            pred_bin = (pred_c > self.threshold)
            mask_bin = (mask_c > self.threshold)

            # Intersection and Union
            tp = (pred_bin & mask_bin).sum().float()
            pred_sum = pred_bin.sum().float()
            mask_sum = mask_bin.sum().float()

            # Compute coefficient
            if self.method == 'iou':
                union = pred_sum + mask_sum - tp
                coef = (tp + self.epsilon) / (union + self.epsilon)
            elif self.method == 'dice':
                denom = pred_sum + mask_sum
                coef = (2 * tp + self.epsilon) / (denom + self.epsilon)
            else:
                raise ValueError(f"Unknown method '{self.method}', choose 'dice' or 'iou'.")
            
            coefs.append(coef)
        return torch.stack(coefs).mean()
    
    def get_loss(self, preds: Tensor, masks: Tensor) -> Tensor:
        """
        Args:
            preds (Tensor): The predicted outputs
            masks (Tensor): The ground truth masks
        """

        # Get coefficient
        coef = self.get_coefficient(preds, masks)

        # Compute loss
        return 1.0 - coef

