import os
import numpy as np
import random
import torch
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Tensor


# MaskDatasetGenerator can be used to generate masks from label files (only 1 class. will be extended to multi-class in the future).
class MaskDatasetGenerator():
    def __init__(
            self,
            label_dir: str,
            mask_dir: str,
            mask_size: Tuple[int, int],
            mask_extension: str,
            mask_fill: int = 255
    ) -> None:
        """
        Args:
            label_dir (str): Path to directory containing label files
            mask_dir (str): Path to directory to save masks
            mask_size (tuple): Size of mask to generate
            mask_extension (str): Extension of mask files
            mask_fill (int): Value to fill mask with (default: 255)
        """

        # Attributes
        self.label_dir = label_dir
        self.mask_dir = mask_dir
        self.mask_size = mask_size
        self.mask_extension = mask_extension
        self.mask_fill = mask_fill

        # Process mask extension
        if not mask_extension.startswith('.'):
            self.mask_extension = '.' + mask_extension

        # Create directories
        if not os.path.exists(self.mask_dir):
            os.makedirs(self.mask_dir, exist_ok=True)

    def generate(self) -> None:
        # Search for label files
        files = [file for file in os.listdir(self.label_dir) if file.endswith('.txt')]

        for label in tqdm(files, desc='Generating masks'):
            label_dir = os.path.join(self.label_dir, label)

            # Read label file
            with open(label_dir, 'r') as file:
                coordinates = [line.strip().split(' ') for line in file.readlines()]

            # Create mask
            zeros = np.zeros(self.mask_size, dtype=np.uint8)
            mask = Image.fromarray(zeros, mode='L')
            draw = ImageDraw.Draw(mask)

            # Draw polygons
            for coords in coordinates:
                if not coords:
                    continue
                try:
                    polygon = [
                        (
                            int(float(coords[i]) * self.mask_size[0]),
                            int(float(coords[i + 1]) * self.mask_size[1])
                        )
                        for i in range(1, len(coords), 2)
                    ]
                    draw.polygon(polygon, fill=self.mask_fill)
                except (IndexError, ValueError) as e:
                    print(f"Error processing coordinates in {label}: {e}")
                    continue

            # Save mask
            label_filename = os.path.basename(label)
            label_name = os.path.splitext(label_filename)[0]
            save_dir = os.path.join(self.mask_dir, label_name + self.mask_extension)
            mask.save(save_dir)


class Augmentation():
    def __init__(
        self,
        channels: int,
        resize: Optional[Tuple[int, int]] = None,
        hflip: bool = False,
        vflip: bool = False,
        rotate: float = 0.0,
        saturation: float = 0.0,
        brightness: float = 0.0,
        factor: float = 1.0,
        p: float = 0.5
    ) -> None:
        """
        Args:
            channels (int): Number of channels in image
            resize (tuple): Resize dimensions (default: None)
            hflip (bool): Whether to flip horizontally (default: False)
            vflip (bool): Whether to flip vertically (default: False)
            rotate (float): Rotation angle (default: 0.0)
            saturation (float): Saturation factor (default: 0.0)
            brightness (float): Brightness factor (default: 0.0)
            factor (float): Factor for brightness and saturation (default: 1.0)
            p (float): Probability of applying transformations (default: 0.5)
        """
        
        # Attributes
        self.channels = channels
        self.resize = resize
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate
        self.saturation = saturation
        self.brightness = brightness
        self.factor = factor
        self.p = p

    def __call__(
            self,
            image: Image.Image,
            mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Args:
            image (PIL.Image): Image to augment
            mask (PIL.Image): Mask to augment
        """

        # Ensure modes
        image = image.convert('L') if self.channels == 1 else image.convert('RGB')
        mask = mask.convert('L')
        
        # Resize
        if self.resize:
            image = F.resize(image, self.resize)
            mask = F.resize(mask, self.resize)
        # Random decisions
        if self.hflip and random.random() < self.p:
            image = F.hflip(image)
            mask = F.hflip(mask)
        if self.vflip and random.random() < self.p:
            image = F.vflip(image)
            mask = F.vflip(mask)
        if self.rotate > 0 and random.random() < self.p:
            angle = random.uniform(-self.rotate, self.rotate)
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)

        # Apply color transformations
        if self.channels == 3:
            if self.saturation > 0:
                image = F.adjust_saturation(image, random.uniform(1 - self.saturation, 1 + self.saturation)*self.factor)
            if self.brightness > 0:
                image = F.adjust_brightness(image, random.uniform(1 - self.brightness, 1 + self.brightness)*self.factor)
        return image, mask


class SegmentationDataset(Dataset):
    def __init__(
            self,
            image_dir: str,
            mask_dir: str,
            extension: str,
            num_images: int = 0,
            augmentation: Optional[Augmentation] = None
    ) -> None:
        """
        Args:
            image_dir (str): Path to directory containing images
            mask_dir (str): Path to directory containing masks
            extension (str): Extension of image and mask files
            num_images (int): Maximum number of images to load (default: 0)
            augmentation (Augmentation): Augmentation to apply (default: None)
        """
        
        # Attributes
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.extension = extension
        self.num_images = num_images
        self.augmentation = augmentation

        # Get all image and mask files
        self.dir = self._get_dir(
            self._get_files(self.image_dir),
            self._get_files(self.mask_dir)
        )
        # Limit number of images
        if self.num_images > 0:
            self.dir['images'] = self.dir['images'][:self.num_images]
            self.dir['masks'] = self.dir['masks'][:self.num_images]

    def _get_dir(self, images: List[str], masks: List[str]) -> Dict[str, List[str]]:
        """
        Args:
            images (list): List of image files
            masks (list): List of mask files
        """

        dir = {
            'images': [],
            'masks': []
        }
        common = set(images).intersection(masks)

        for file in common:
            dir['images'].append(os.path.join(self.image_dir, file))
            dir['masks'].append(os.path.join(self.mask_dir, file))
        return dir
    
    def _get_files(self, dir: str) -> List[str]:
        """
        Args:
            dir (str): Directory to search for files
        """

        return [file for file in os.listdir(dir) if file.endswith(self.extension)]
    
    def __len__(self) -> int:
        return len(self.dir['images'])
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            idx (int): Index of the item to get
        """
        
        image_dir = self.dir['images'][idx]
        mask_dir = self.dir['masks'][idx]

        # Open image and mask
        try:
            image = Image.open(image_dir)
            mask = Image.open(mask_dir)
        except Exception as e:
            raise IOError(f"Error opening {image_dir} or {mask_dir}: {e}")
        
        # Augmentation
        if self.augmentation:
            image, mask = self.augmentation(image, mask)
        
        # Convert to tensor
        if not isinstance(image, Tensor):
            image = F.to_tensor(image)
        if not isinstance(mask, Tensor):
            mask = F.to_tensor(mask)
        return image, mask


class SegmentationDataLoader():
    def __init__(
            self,
            dataset: SegmentationDataset,
            split: Dict[str, float] = {'train': 0.8, 'valid': 0.1, 'test': 0.1},
            batch_size: int = 8,
            shuffle: bool = True,
            num_workers: int = 0,
            pin_memory: bool = True,
            seed: int = 42
    ) -> None:
        """
        Args:
            dataset (SegmentationDataset): Dataset to load
            split (dict): Split proportions for train, valid, and test sets (default: {'train': 0.8, 'valid': 0.1, 'test': 0.1})
            batch_size (int): Batch size (default: 8)
            shuffle (bool): Whether to shuffle the dataset (default: True)
            num_workers (int): Number of worker threads (default: 0)
            pin_memory (bool): Whether to use pin memory (default: True)
            seed (int): Random seed (default: 42)
        """
        
        # Attributes
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

    def get_loaders(self) -> Dict[str, DataLoader]:
        # Get number of images in each split
        total = len(self.dataset)
        n_train = int(self.split['train'] * total)
        n_valid = int(self.split['valid'] * total)
        n_test = total - n_train - n_valid

        # Split dataset into 'train', 'valid', 'test'
        splits = random_split(
            dataset=self.dataset,
            lengths=[n_train, n_valid, n_test],
            generator=torch.Generator().manual_seed(self.seed)
        )

        # Create loaders
        loaders = {}
        phases = ['train', 'valid', 'test']
        for phase, dataset in zip(phases, splits):
            loaders[phase] = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=(self.shuffle and phase == 'train'), # Only shuffle training dataset
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=(phase == 'train')                 # Only drop last incomplete batch during training
            )

        print(f"Dataset split: train={n_train}, valid={n_valid}, test={n_test} samples.")
        return loaders

