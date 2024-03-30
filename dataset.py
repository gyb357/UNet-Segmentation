from torch import device
from PIL import Image
from torchvision.transforms import functional as F
import random
from torch.utils.data import Dataset
import os


class Augmentation():
    def __init__(
            self,
            channels: int = None,
            resize: tuple = None,
            crop: tuple = None,
            hflip: bool = False,
            vflip: bool = False,
            p: float = 0.5,
            device: device = None,
            gpu: bool = False
    ):
        self.channels = channels
        self.resize = resize
        self.crop = crop
        self.hflip = hflip
        self.vflip = vflip
        self.p = p
        self.device = device
        self.gpu = gpu

    def __call__(self, origin: Image, mask: Image):
        if self.channels == 1:
            origin = origin.convert('L')
            mask = mask.convert('1')
        if self.resize is not None:
            origin = F.resize(origin, self.resize)
            mask = F.resize(mask, self.resize)
        if self.crop is not None:
            origin = F.center_crop(origin, self.crop)
            mask = F.center_crop(mask, self.crop)
        if self.hflip == True:
            if random.random() < self.p:
                origin = F.hflip(origin)
                mask = F.hflip(mask)
        if self.vflip == True:
            if random.random() < self.p:
                origin = F.vflip(origin)
                mask = F.vflip(mask)
        origin = F.to_tensor(origin)
        mask = F.to_tensor(mask)
        if self.gpu == True:
            origin.to(self.device)
            mask.to(self.device)
        return origin, mask


class SegmentationDataset(Dataset):
    def __init__(self, origin_path: str, mask_path: str, extension: str, num_images: int, augmentation: Augmentation):
        self.origin_path = origin_path
        self.mask_path = mask_path
        self.extension = extension
        self.num_images = num_images
        self.augmentation = augmentation
        self.path = self.get_path(self.get_files(origin_path), self.get_files(mask_path))

    def get_path(self, origins: list, masks: list) -> dict:
        paths = {
            'origin': [],
            'mask': []
        }
        for mask in masks:
            if mask in origins:
                paths['origin'].append(os.path.join(self.origin_path, mask))
                paths['mask'].append(os.path.join(self.mask_path, mask))
        return paths
    
    def get_files(self, path: str) -> list:
        return [f for f in os.listdir(path) if f.endswith(self.extension)]
    
    def __getitem__(self, idx):
        if self.num_images:
            idx = idx % len(self.path['origin'])

        origin_path = self.path['origin'][idx]
        mask_path = self.path['mask'][idx]

        with open(origin_path, 'rb') as f_origin, open(mask_path, 'rb') as f_mask:
            origin = Image.open(f_origin)
            mask = Image.open(f_mask)

            origin, mask = self.augmentation(origin, mask)
            return origin, mask

    
    def __len__(self):
        if self.num_images:
            return int(min(self.num_images, len(self.path['origin'])))
        else:
            return len(self.path['origin'])

