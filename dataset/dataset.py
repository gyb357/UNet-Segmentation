import logging
import os
import numpy as np
import torch.nn.functional as F
import random
import torch
from typing import Tuple, Optional, List, Dict
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Tensor


class MaskDatasetGenerator():
    """Generate a mask image with the Yolo dataset format as input."""

    def __init__(
            self,
            label_path: str,
            mask_path: str,
            mask_size: Tuple[int, int],
            mask_extension: str,
            mask_fill: int = 255
    ) -> None:
        """
        Args:
            label_path: Path to the directory containing label files
            mask_path: Path where mask images will be saved
            mask_size: Size of the output mask (width, height)
            mask_extension: File extension for saved masks (e.g., '.png')
            mask_fill: Pixel value for the foreground (default: 255)
        """
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        # Make directory
        os.makedirs(mask_path, exist_ok=True)

        # Attributes
        self.logger = logging.getLogger(__name__)
        self.label_path = label_path
        self.mask_path = mask_path
        self.mask_size = mask_size
        self.mask_extension = mask_extension
        self.mask_fill = mask_fill

    def __call__(self) -> None:
        label_files = [file for file in os.listdir(self.label_path) if file.endswith('.txt')]

        for i, label in enumerate(label_files):
            label_file_path = os.path.join(self.label_path, label)

            with open(label_file_path, 'r') as file:
                coordinates = [line.strip().split(' ') for line in file.readlines()]

            zeros = np.zeros(self.mask_size, dtype=np.uint8)
            mask = Image.fromarray(zeros, mode='L')
            draw = ImageDraw.Draw(mask)

            for coords in coordinates:
                if not coords:
                    continue

                try:
                    polygon = [
                        (
                            int(float(coords[i])*self.mask_size[0]),
                            int(float(coords[i + 1])*self.mask_size[1])
                        )
                        for i in range(1, len(coords), 2)
                    ]
                    draw.polygon(polygon, fill=self.mask_fill)
                except (IndexError, ValueError) as e:
                    self.logger.error(f"Error processing coordinates in {label}: {e}")
                    continue

            save_path = os.path.join(self.mask_path, label.replace('.txt', '') + self.mask_extension)
            mask.save(save_path)

            if (i + 1)%100 == 0:
                self.logger.info(f"Processed {i + 1}/{len(label_files)} mask files")

        self.logger.info(f"Mask generation completed. {len(label_files)} masks generated.")


class Augmentation():
    """Class for applying various image augmentations to image and mask pairs."""

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
            channels: Number of channels in input images (1 for grayscale, 3 for RGB)
            resize: Optional tuple for resizing (width, height)
            hflip: Whether to apply horizontal flip
            vflip: Whether to apply vertical flip
            rotate: Maximum rotation angle (will use random angle between -rotate and +rotate)
            saturation: Maximum saturation adjustment factor
            brightness: Maximum brightness adjustment factor
            factor: Global adjustment scale factor
            p: Probability of applying each augmentation
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
        # Convert to grayscale if needed
        if self.channels == 1:
            image = F.to_grayscale(image, 1)

        # Ensure mask is binary
        mask = mask.convert('1')

        # Apply geometric transformations
        if self.resize is not None:
            image = F.resize(image, self.resize)
            mask = F.resize(image, self.resize)
        if self.hflip and random.random() < self.p:
            image = F.hflip(image)
            mask = F.hflip(image)
        if self.vflip and random.random() < self.p:
            image = F.vflip(image)
            mask = F.vflip(image)
        if self.rotate > 0:
            angle = random.uniform(-self.rotate, self.rotate)
            image = F.rotate(image, angle)
            mask = F.rotate(image, angle)

        # Apply color transformations (to image only)
        if self.saturation > 0:
            image = F.adjust_saturation(image, random.uniform(1 - self.saturation, 1 + self.saturation)*self.factor)
        if self.brightness > 0:
            image = F.adjust_brightness(image, random.uniform(1 - self.brightness, 1 + self.brightness)*self.factor)
        return image, mask


class SegmentationDataset(Dataset):
    """Dataset class for loading image-mask pairs for segmentation tasks."""

    def __init__(
            self,
            image_path: str,
            mask_path: str,
            extension: str,
            num_images: int = 0,
            augmentation: Optional[Augmentation] = None
    ) -> None:
        """
        Args:
            image_path: Directory containing input images
            mask_path: Directory containing corresponding mask images
            extension: File extension to filter for (e.g., '.png')
            num_images: Maximum number of images to use (0 means use all)
            augmentation: Optional augmentation object to apply
        """

        # Attributes
        self.image_path = image_path
        self.mask_path = mask_path
        self.extension = extension
        self.num_images = num_images
        self.augmentation = augmentation

        # Get file paths
        self.path = self._get_paths(self._get_files(image_path), self._get_files(mask_path))
        print(f"Found {len(self.path['image'])} image-mask pairs")

    def _get_paths(self, images: List[str], masks: List[str]) -> Dict[str, List[str]]:
        path = {
            'image': [],
            'mask': []
        }

        # Find common files
        common_files = set(images).intersection(masks)

        # Append common files to paths
        for file in common_files:
            path['image'].append(os.path.join(self.image_path, file))
            path['mask'].append(os.path.join(self.mask_path, file))
        return path
    
    def _get_files(self, path: str) -> List[str]:
        return [file for file in os.listdir(path) if file.endswith(self.extension)]
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # Load images
        try:
            image = Image.open(self.path['image'][idx])
            mask = Image.open(self.path['mask'][idx])
        except Exception as e:
            raise IOError(f"Error loading images at index {idx}: {e}")
        
        # Apply augmentations
        if self.augmentation:
            image, mask = self.augmentation(image, mask)

        # Convert to tensors
        if not isinstance(image, Tensor):
            image = F.to_tensor(image)
        if not isinstance(mask, Tensor):
            mask = F.to_tensor(mask)
        return image, mask
    
    def __len__(self) -> int:
        image_len = len(self.path['image'])

        if self.num_images > 0 and self.num_images <= image_len:
            return self.num_images
        return image_len


class SegmentationDataLoader(DataLoader):
    """DataLoader class for segmentation datasets and managing train/val/test splits."""

    default_split: Dict[str, float] = {
        'train': 0.8,
        'val': 0.1,
        'test': 0.1
    }

    def __init__(
            self,
            dataset: SegmentationDataset,
            dataset_split: Optional[Dict[str, float]] = None,
            batch_size: int = 8,
            shuffle: bool = True,
            num_workers: int = 0,
            pin_memory: bool = False
    ) -> None:
        """
        Args:
            dataset: SegmentationDataset instance
            dataset_split: Dictionary with split ratios (train, val, test)
            batch_size: Batch size for dataloaders
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
        """

        # Attributes
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Initialize dataset split with defaults if not provided
        self.dataset_split = dataset_split or self.default_split

    def _get_length(self) -> Tuple[int, int, int]:
        dataset_len = len(self.dataset)
        train = int(self.dataset_split['train']*dataset_len)
        val = int(self.dataset_split['val']*dataset_len)
        test = dataset_len - train - val
        return train, val, test
    
    def get_loader(self, seed: int = 42) -> Dict[str, DataLoader]:
        generator = torch.Generator().manual_seed(seed)

        # Get split lengths
        train_len, val_len, test_len = self._get_length()
        # Split the dataset
        splits = random_split(self.dataset, [train_len, val_len, test_len], generator)

        # Create dataloaders
        loaders = {}
        phases = ['train', 'val', 'test']

        for phase, dataset in zip(phases, splits):
            loaders[phase] = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=(self.shuffle and phase == 'train'), # Only shuffle training dataset
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=(phase == 'train')                 # Only drop last incomplete batch during training
            )

        print(f"Dataset split: train={train_len}, val={val_len}, test={test_len} samples")
        return loaders

