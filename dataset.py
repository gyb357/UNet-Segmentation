from typing import Optional, Tuple, List, Dict
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
            mask_extension: str,
            mask_fill: int = 255
    ) -> None:
        self.label_path = label_path
        self.mask_path = mask_path
        self.mask_size = mask_size
        self.mask_extension = mask_extension
        self.mask_fill = mask_fill
        os.makedirs(mask_path, exist_ok=True)

    def __call__(self) -> None:
        for label in os.listdir(self.label_path):
            label_files = os.path.join(self.label_path, label)

            with open(label_files, 'r') as file:
                coordinates = [line.strip().split(' ') for line in file.readlines()]

            zeros = np.zeros(self.mask_size, dtype=np.uint8)
            mask = Image.fromarray(zeros, mode='L')
            draw = ImageDraw.Draw(mask)

            for coords in coordinates:
                polygon = [
                    (
                        int(float(coords[i])*self.mask_size[0]),
                        int(float(coords[i + 1])*self.mask_size[1])
                    )
                    for i in range(1, len(coords), 2)
                ]
                draw.polygon(polygon, fill=self.mask_fill)

            save_path = os.path.join(self.mask_path, label.replace('.txt', '') + self.mask_extension)
            mask.save(save_path)


class Augmentation():
    norm_mean: List[float] = [0.485, 0.456, 0.406]
    norm_std: List[float] = [0.229, 0.224, 0.225]

    def __init__(
            self,
            channels: int,
            resize: Optional[Tuple[int, int]] = None,
            hflip: bool = False,
            vflip: bool = False,
            rotate: float = 0,
            saturation: float = 0,
            brightness: float = 0,
            factor: float = 1,
            p: float = 0.5,
            normalize: bool = False
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
        self.normalize = normalize

    def __call__(self, image: Image, mask: Image) -> Tuple[Image.Image, Image.Image]:
        if self.channels == 1:
            image = F.to_grayscale(image, 1)

        mask = F.to_tensor(mask)/255.0
        mask = F.to_pil_image(mask)

        if self.resize is not None:
            image, mask = F.resize(image, self.resize), F.resize(mask, self.resize)
        if self.hflip and R.random() < self.p:
            image, mask = F.hflip(image), F.hflip(mask)
        if self.vflip and R.random() < self.p:
            image, mask = F.vflip(image), F.vflip(mask)
        if self.rotate > 0:
            angle = R.uniform(-self.rotate, self.rotate)
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)

        if self.saturation > 0:
            image = F.adjust_saturation(image, R.uniform(1 - self.saturation, 1 + self.saturation)*self.factor)
        if self.brightness > 0:
            image = F.adjust_brightness(image, R.uniform(1 - self.brightness, 1 + self.brightness)*self.factor)

        if self.normalize:
            image = F.to_tensor(image)
            mask = F.to_tensor(mask)
            image = F.normalize(image, self.norm_mean, self.norm_std)
            mask = F.normalize(mask, self.norm_mean, self.norm_std)
        return image, mask


class SegmentationDataLoader(Dataset):
    def __init__(
            self,
            image_path: str,
            mask_path: str,
            extension: str,
            num_images: int = 0,
            augmentation: Optional[Augmentation] = None
    ) -> None:
        self.image_path = image_path
        self.mask_path = mask_path
        self.extension = extension
        self.num_images = num_images
        self.augmentation = augmentation
        self.path = self.get_path(self.get_file(image_path), self.get_file(mask_path))

    def get_path(self, images: list, masks: list) -> Dict[str, List[str]]:
        path = {
            'image': [],
            'mask': []
        }
        for mask in masks:
            if mask in images:
                path['image'].append(os.path.join(self.image_path, mask))
                path['mask'].append(os.path.join(self.mask_path, mask))
        return path
    
    def get_file(self, path: str) -> List[str]:
        return [f for f in os.listdir(path) if f.endswith(self.extension)]
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image = Image.open(self.path['image'][idx])
        mask = Image.open(self.path['mask'][idx])

        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)

        if not isinstance(image, Tensor):
            image = F.to_tensor(image)
        if not isinstance(mask, Tensor):
            mask = F.to_tensor(mask)
        return image, mask
    
    def __len__(self) -> int:
        if self.num_images != 0 and self.num_images <= len(self.path['image']):
            return self.num_images
        return len(self.path['image'])


class SegmentationDataset():
    default_split: Dict[str, float] = {'train': 0.8, 'val': 0.1}

    def __init__(
            self,
            dataset_loader: SegmentationDataLoader,
            dataset_split: Dict[str, float],
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

        if 'train' not in dataset_split:
            self.dataset_split['train'] = self.default_split['train']
        if 'val' not in dataset_split:
            self.dataset_split['val'] = self.default_split['val']

    def get_length(self) -> Tuple[int, int, int]:
        dataset_len = len(self.dataset_loader)
        train = int(self.dataset_split['train']*dataset_len)
        val = int(self.dataset_split['val']*dataset_len)
        test = dataset_len - train - val
        return train, val, test
    
    def get_loader(self, debug: bool = False) -> Dict[str, DataLoader]:
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

