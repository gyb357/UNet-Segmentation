from numpy import ndarray
import matplotlib.pyplot as plt
from typing import Type, Union
from unet_module import EnsembleUNet, UNet
from torch import device
from dataset import Augmentation
from PIL import Image
from torchvision.transforms import functional as F
import torch
from utils import tensor_to_numpy


def show_image(input: ndarray, output: ndarray, result: ndarray) -> None:
    plt.subplot(1, 3, 1)
    plt.title('Origin')
    plt.imshow(input.permute(1, 2, 0))

    plt.subplot(1, 3, 2)
    plt.title('Predicted mask')
    plt.imshow(output.squeeze())

    plt.subplot(1, 3, 3)
    plt.title('Result')
    plt.imshow(result.squeeze().transpose(1, 2, 0))
    plt.show()


class Tester():
    def __init__(
            self,
            model: Type[Union[EnsembleUNet, UNet]],
            device: device,
            augmentation: Augmentation,
            threshold: float
    ) -> None:
        self.model = model
        self.device = device
        self.augmentation = augmentation
        self.threshold = threshold

    def test(self, model_path: str, image_path: str) -> None:
        input = Image.open(image_path)

        if self.augmentation.channels == 1:
            input = input.convert('L')

        input = F.resize(input, self.augmentation.resize)
        input = F.to_tensor(input)
        input_copy = input

        input = input.to(self.device)
        input = input.unsqueeze(0)

        self.model.load_state_dict(torch.load(model_path))
        with torch.no_grad():
            output = self.model(input)

        input = tensor_to_numpy(input)
        output = tensor_to_numpy(torch.sigmoid(output)) # output = tensor_to_numpy(torch.softmax(output))
        result = (output > self.threshold)*input
        show_image(input_copy, output, result)

