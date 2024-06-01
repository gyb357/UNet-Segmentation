from unet import UNet
import torch
from dataset import Augmentation
from numpy import ndarray
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as F
from torchvision import transforms
import torch
from utils import tensor_to_numpy


class Test:
    def __init__(
            self,
            model: UNet,
            device: torch.device,
            augmentation: Augmentation,
            threshold: float,
            model_path: str,
            image_path: str
    ) -> None:
        self.model = model
        self.device = device
        self.augmentation = augmentation
        self.threshold = threshold
        self.model_path = model_path
        self.image_path = image_path

        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def show_image(self, input: ndarray, output: ndarray, result: ndarray) -> None:
        plt.subplot(1, 3, 1)
        plt.title('Original')
        plt.imshow(input.permute(1, 2, 0))

        plt.subplot(1, 3, 2)
        plt.title('Predicted Mask')
        plt.imshow(output.squeeze(), cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title('Result')
        plt.imshow(result.squeeze().transpose(1, 2, 0))
        plt.show()

    def test(self) -> None:
        input_image = Image.open(self.image_path)

        if self.augmentation.channels == 1:
            input_image = input_image.convert('L')
        input_image = F.resize(input_image, self.augmentation.resize)
        input_tensor = transforms.ToTensor()(input_image).to(self.device).unsqueeze(0)

        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        input_np = tensor_to_numpy(input_tensor)
        output_np = tensor_to_numpy(output_tensor)
        result_np = (output_np > self.threshold)*input_np

        self.show_image(input_tensor.squeeze(0), output_np, result_np)

