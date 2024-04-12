from model import UNet
from torch import device, Tensor
from utils import tensor_to_numpy
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import transforms
import torch


class Test():
    def __init__(
            self,
            model: UNet,
            device: device,
            threshold: float,
            model_path: str,
            image_path: str
    ) -> None:
        self.model = model
        self.device = device
        self.threshold =threshold
        self.model_path = model_path
        self.image_path = image_path

    def show_image(self, input: Tensor, output: Tensor, result: Tensor) -> None:
        plt.subplot(1, 3, 1)
        plt.title('Origin')
        plt.imshow(input.squeeze())

        plt.subplot(1, 3, 2)
        plt.title('Predicted mask')
        plt.imshow(output.squeeze())

        plt.subplot(1, 3, 3)
        plt.title('Result')
        plt.imshow(result.squeeze())
        plt.show()

    def test(self) -> None:
        input = Image.open(self.image_path).convert('L')
        input = F.resize(input, (256, 256))
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

