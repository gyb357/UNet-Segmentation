import torch
from torch import Tensor


class Loss():
    def __init__(
            self,
            num_classes: int,
            threshold: float = 0.5,
            epsilon: float = 1e-6,
    ) -> None:
        """
        Args:
            num_classes (int): The number of classes.
            epsilon (float): The epsilon value. (default: 1e-6)
            threshold (float): The threshold value. (default: 0.5)
        """

        # Attributes
        self.num_classes = num_classes
        self.threshold = threshold
        self.epsilon = epsilon
        self.activation = torch.sigmoid if num_classes == 1 else lambda x: torch.softmax(x, dim=1)

    def get_coefficient(self, outputs: Tensor, masks: Tensor, method: str) -> Tensor:
        """
        Args:
            outputs (Tensor): The predicted outputs.
            masks (Tensor): The ground truth masks.
            method (str): The method to compute the coefficient. ('dice' or 'iou')
        """

        # Activation function
        preds = self.activation(outputs)

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
            if method == 'iou':
                union = pred_sum + mask_sum - tp
                coef = (tp + self.epsilon) / (union + self.epsilon)
            elif method == 'dice':
                denom = pred_sum + mask_sum
                coef = (2 * tp + self.epsilon) / (denom + self.epsilon)
            else:
                raise ValueError(f"Unknown method '{method}', choose 'dice' or 'iou'.")
            
            coefs.append(coef)
        return torch.stack(coefs).mean()
    
    def get_loss(self, outputs: Tensor, masks: Tensor, method: str) -> Tensor:
        """
        Args:
            outputs (Tensor): The predicted outputs.
            masks (Tensor): The ground truth masks.
            method (str): The method to compute the coefficient. ('dice' or 'iou')
        """

        # Get coefficient
        coef = self.get_coefficient(outputs, masks, method)

        # Compute loss
        return 1.0 - coef

