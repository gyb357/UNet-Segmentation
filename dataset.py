from PIL import Image
from torchvision.transforms import functional as F
import random
from torch.utils.data import Dataset, random_split, DataLoader
import os


class Augmentation():
    def __init__(
            self,
            channels: int = None,
            crop: list = None,
            resize: list = None,
            hflip: bool = False,
            vflip: bool = False,
            p: float = 0.5
    ) -> None:
        self.channels = channels
        self.crop = crop
        self.resize = resize
        self.hflip = hflip
        self.vflip = vflip
        self.p = p

    def __call__(self, image: Image, mask: Image):
        data = [image, mask]

        # Convert image to grayscale, binary
        if self.channels == 1:
            data[0] = data[0].convert('L')
        data[1] = data[1].convert('1')

        # Commonly applied augmentations
        for i, img in enumerate(data):
            if isinstance(self.crop, list):
                img = F.center_crop(img, self.crop)
            if isinstance(self.resize, list):
                img = F.resize(img, self.resize)
            data[i] = img

        # Horizon, vertical flip
        if self.hflip and random.random() < self.p:
            data[0] = F.hflip(data[0])
            data[1] = F.hflip(data[1])
        if self.vflip and random.random() < self.p:
            data[0] = F.vflip(data[0])
            data[1] = F.vflip(data[1])
        return data[0], data[1]


class SegmentationDataset(Dataset):
    def __init__(
            self,
            image_path: str,
            mask_path: str,
            extension: str,
            num_images: int = 0,
            augmentation = None
    ) -> None:
        self.image_path = image_path
        self.mask_path = mask_path
        self.extension = extension
        self.num_images = num_images
        self.augmentation = augmentation
        self.paths = self.get_path(self.get_files(image_path), self.get_files(mask_path))

    def get_path(self, images: list, masks: list) -> dict:
        paths = {
            'image': [],
            'mask': []
        }
        for mask in masks:
            if mask in images:
                paths['image'].append(os.path.join(self.image_path, mask))
                paths['mask'].append(os.path.join(self.mask_path, mask))
        return paths
    
    def get_files(self, path: str) -> list:
        return [f for f in os.listdir(path) if f.endswith(self.extension)]
    
    def __getitem__(self, index: int) -> tuple:
        # Import images
        image = Image.open(self.paths['image'][index])
        mask = Image.open(self.paths['mask'][index])

        # Apply augmentations
        if isinstance(self.augmentation, Augmentation):
            image, mask = self.augmentation(image, mask)
        return F.to_tensor(image), F.to_tensor(mask)
    
    def __len__(self) -> int:
        if self.num_images != 0:
            return self.num_images
        else:
            return len(self.paths['image'])


class Dataset():
    def __init__(
            self,
            dataset: SegmentationDataset,
            dataset_split: dict,
            batch_size: int,
            shuffle: bool,
            num_workers: int,
            pin_memory: bool
    ):
        self.dataset = dataset
        self.dataset_split = dataset_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @staticmethod
    def dataset_lengths(dataset: SegmentationDataset, split: dict):
        dataset_len = len(dataset)
        train_len = int(split['train']*dataset_len)
        val_len = int(split['val']*dataset_len)
        test_len = dataset_len - train_len - val_len
        return train_len, val_len, test_len
    
    def get_dataloader(self) -> dict:
        train_len, val_len, test_len = self.dataset_lengths(self.dataset, self.dataset_split)
        train_set, val_set, test_set = random_split(self.dataset, [train_len, val_len, test_len])

        loaders = {}
        for phase, dataset in {'train': train_set, 'val': val_set, 'test': test_set}.items():
            loaders[phase] = DataLoader(
                dataset,
                self.batch_size,
                self.shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
        print(f'train_loader: {train_len}, valid_loader: {val_len}, test_loader: {test_len}')
        return loaders

