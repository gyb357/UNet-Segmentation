from typing import Tuple, Dict
from utils import makedirs
import os
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
import random as R
from torch.utils.data import Dataset, random_split, DataLoader
from torch import Tensor


class MaskDatasetGenerator():
    def __init__(
            self,
            label_path: str,
            mask_path: str,
            mask_size: Tuple[int, int],
            mask_extension: str
    ) -> None:
        self.label_path = label_path
        self.mask_path = mask_path
        self.mask_size = mask_size
        self.mask_extension = mask_extension
        makedirs(mask_path)

    def __call__(self) -> None:
        for label in os.listdir(self.label_path):
            labels = open(os.path.join(self.label_path, label))

            coordinate = []
            for line in labels.readlines():
                line.replace('\n', '')
                coordinate.append(line.split(' '))

            zeros = np.zeros(self.mask_size, dtype=np.uint8)
            mask = Image.fromarray(zeros.squeeze(), mode='L')
            draw = ImageDraw.Draw(mask)

            for coords in coordinate:
                polygon = []
                for i in range(1, len(coords), 2):
                    x = int(float(coords[i])*self.mask_size[0])
                    y = int(float(coords[i + 1])*self.mask_size[1])
                    polygon.append((x, y))
                draw.polygon(polygon, fill=255)

            save_path = os.path.join(self.mask_path, label.replace('.txt', '') + self.mask_extension)
            mask.save(save_path)


class Augmentation():
    def __init__(
            self,
            channels: int,
            resize: Tuple[int, int] = None,
            hflip: bool = False,
            vflip: bool = False,
            rotate: float = 0,
            saturation: float = 0,
            brightness: float = 0,
            factor: float = 1,
            p: float = 0.5
    ) -> None:
        self.channels = channels
        self.resize = resize
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate
        self.saturation = saturation
        self.brightness = brightness
        self.factor = factor
        self.p = p

    def __call__(self, image: Image, mask: Image) -> Tuple[Image.Image, Image.Image]:
        if self.channels == 1:
            image = F.to_grayscale(image, 1)
        image = F.adjust_saturation(image, R.uniform(1 - self.saturation, 1 + self.saturation)*self.factor)
        image = F.adjust_brightness(image, R.uniform(1 - self.brightness, 1 + self.brightness)*self.factor)

        mask = F.to_tensor(mask)/255.0
        mask = F.to_pil_image(mask)

        hflip_p = R.random()
        vflip_p = R.random()
        angle = R.randrange(-self.rotate, self.rotate)

        data = [image, mask]
        for i, img in enumerate(data):
            if self.resize is not None:
                img = F.resize(img, self.resize)
            if self.hflip and hflip_p < self.p:
                img = F.hflip(img)
            if self.vflip and vflip_p < self.p:
                img = F.vflip(img)
            if self.rotate != 0:
                img = F.rotate(img, angle)
            data[i] = img
        return data[0], data[1]


class SegmentationDataLoader(Dataset):
    def __init__(
            self,
            image_path: str,
            mask_path: str,
            extension: str,
            num_images: int = 0,
            augmentation: Augmentation = None
    ) -> None:
        self.image_path = image_path
        self.mask_path = mask_path
        self.extension = extension
        self.num_images = num_images
        self.augmentation = augmentation
        self.path = self.get_path(self.get_file(image_path), self.get_file(mask_path))

    def get_path(self, images: list, masks: list) -> Dict[list, list]:
        path = {
            'image': [],
            'mask': []
        }
        for mask in masks:
            if mask in images:
                path['image'].append(os.path.join(self.image_path, mask))
                path['mask'].append(os.path.join(self.mask_path, mask))
        return path
    
    def get_file(self, path: str) -> list:
        return [f for f in os.listdir(path) if f.endswith(self.extension)]
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image = Image.open(self.path['image'][idx])
        mask = Image.open(self.path['mask'][idx])

        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)

        image = F.to_tensor(image)
        mask = F.to_tensor(mask)
        return image, mask
    
    def __len__(self) -> int:
        if self.num_images != 0 and self.num_images <= self.path['image']:
            return self.num_images
        else: return len(self.path['image'])


class SegmentationDataset():
    def __init__(
            self,
            dataset_loader: SegmentationDataLoader,
            dataset_split: dict,
            batch_size: int,
            shuffle: bool = False,
            num_workers: int = 0,
            pin_memory: bool = False
    ) -> None:
        self.dataset_loader = dataset_loader
        self.dataset_split = dataset_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def get_length(self) -> Tuple[int, int, int]:
        dataset_len = len(self.dataset_loader)
        train = int(self.dataset_split['train']*dataset_len)
        val = int(self.dataset_split['val']*dataset_len)
        test = dataset_len - train - val
        return train, val, test
    
    def get_loader(self, debug: bool = False) -> dict:
        train_len, val_len, test_len = self.get_length()
        train_set, val_set, test_set = random_split(self.dataset_loader, [train_len, val_len, test_len])

        loader = {}
        for phase, dataset in {'train': train_set, 'val': val_set, 'test': test_set}.items():
            loader[phase] = DataLoader(
                dataset,
                self.batch_size,
                self.shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )

        if debug:
            print(f'train_set: {train_len}, valid_set: {val_len}, test_set: {test_len}')
        return loader

