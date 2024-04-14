from model import UNet
from torch import device
from dataset import Augmentation
from numpy import ndarray
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import transforms
import torch
from utils import tensor_to_numpy


class Test():
    def __init__(
            self,
            model: UNet,
            device: device,
            augmentation: Augmentation,
            threshold: float,
            model_path: str,
            image_path: str
    ) -> None:
        self.model = model
        self.device = device
        self.augmentation = augmentation
        self.threshold =threshold
        self.model_path = model_path
        self.image_path = image_path
        
    def show_image(self, input: ndarray, output: ndarray, result: ndarray) -> None:
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

    def test(self) -> None:
        input = Image.open(self.image_path)

        if self.augmentation.channels == 1:
            input = input.convert('L')
        input = F.resize(input, self.augmentation.resize)
        input = transforms.ToTensor()(input)
        input_copy = input
        
        input = input.to(self.device)
        input = input.unsqueeze(0)

        self.model.load_state_dict(torch.load(self.model_path))
        with torch.no_grad():
            output = self.model(input)

        input = tensor_to_numpy(input)
        output = tensor_to_numpy(torch.sigmoid(output))
        result = (output > self.threshold)*input
        self.show_image(input_copy, output, result)

