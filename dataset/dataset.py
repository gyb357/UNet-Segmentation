import os
import logging
import numpy as np
import random
import torch.nn.functional as F
import torch
from typing import Tuple, Optional, Dict, List
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Tensor


class MaskDatasetGenerator():
    """Generate a mask image with the Yolo v8 dataset format as input."""

    def __init__(
            self,
            label_path: str,
            mask_path: str,
            mask_size: Tuple[int, int],
            mask_extension: str,
            mask_fill: int = 255
    ) -> None:
        """
        Initialize the MaskDatasetGenerator.
        
        Args:
            label_path: Path to the directory containing label files
            mask_path: Path where mask images will be saved
            mask_size: Size of the output mask (width, height)
            mask_extension: File extension for saved masks (e.g., '.png')
            mask_fill: Pixel value for the foreground (default: 255)
        """

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        label_files = [f for f in os.listdir(self.label_path) if f.endswith('.txt')]
        
        if not label_files:
            self.logger.warning(f"No label files found in {self.label_path}")
            return
            
        self.logger.info(f"Found {len(label_files)} label files. Starting mask generation...")
        
        for i, label in enumerate(label_files):
            label_file_path = os.path.join(self.label_path, label)
            
            try:
                with open(label_file_path, 'r') as file:
                    coordinates = [line.strip().split(' ') for line in file.readlines()]

                zeros = np.zeros(self.mask_size, dtype=np.uint8)
                mask = Image.fromarray(zeros, mode='L')
                draw = ImageDraw.Draw(mask)

                for coords in coordinates:
                    # Skip empty lines
                    if not coords:
                        continue
                        
                    try:
                        # Extract polygon points
                        polygon = [
                            (
                                int(float(coords[i])*self.mask_size[0]),
                                int(float(coords[i + 1])*self.mask_size[1])
                            )
                            for i in range(1, len(coords), 2)
                        ]
                        
                        # Check if polygon has enough points
                        if len(polygon) < 3:
                            self.logger.warning(f"Skipping invalid polygon with {len(polygon)} points in {label}")
                            continue
                            
                        draw.polygon(polygon, fill=self.mask_fill)
                    except (IndexError, ValueError) as e:
                        self.logger.error(f"Error processing coordinates in {label}: {e}")
                        continue

                save_path = os.path.join(self.mask_path, label.replace('.txt', self.mask_extension))
                mask.save(save_path)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(label_files)} mask files")
                    
            except Exception as e:
                self.logger.error(f"Error processing {label}: {e}")
                
        self.logger.info(f"Mask generation completed. {len(label_files)} masks generated.")


class Augmentation():
    """Class for applying various image augmentations to image and mask pairs."""
    
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
            p: float = 0.5
    ) -> None:
        """
        Initialize augmentation parameters.
        
        Args:
            channels: Number of channels for the output image (1 for grayscale, 3 for RGB)
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

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Apply augmentations to an image-mask pair.
        
        Args:
            image: Input image
            mask: Corresponding segmentation mask
        """

        # Convert to grayscale if needed
        if self.channels == 1:
            image = F.to_grayscale(image, 1)

        # Ensure mask is binary
        mask = mask.convert('1')

        # Apply geometric transformations
        if self.resize is not None:
            image = F.resize(image, self.resize)
            mask = F.resize(mask, self.resize, interpolation=Image.NEAREST)
            
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


class SegmentationDataLoader(Dataset):
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
        Initialize the segmentation data loader.
        
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
        
        # Validate paths
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask path does not exist: {mask_path}")
            
        # Get file paths
        self.paths = self._get_matched_paths()
        
        if len(self.paths['image']) == 0:
            raise ValueError(f"No matching image-mask pairs found with extension {extension}")

    def _get_matched_paths(self) -> Dict[str, List[str]]:
        """Get matched image and mask file paths."""

        image_files = set(self._get_files(self.image_path))
        mask_files = set(self._get_files(self.mask_path))
        
        # Find common files
        common_files = image_files.intersection(mask_files)
        
        paths = {
            'image': [],
            'mask': []
        }
        
        for file in common_files:
            paths['image'].append(os.path.join(self.image_path, file))
            paths['mask'].append(os.path.join(self.mask_path, file))
            
        return paths
    
    def _get_files(self, path: str) -> List[str]:
        """
        Get list of files with the specified extension in a directory.
        
        Args:
            path: Directory path
        """

        return [f for f in os.listdir(path) if f.endswith(self.extension)]
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # Load images
        try:
            image = Image.open(self.paths['image'][idx]).convert('RGB')
            mask = Image.open(self.paths['mask'][idx]).convert('L')
        except Exception as e:
            raise IOError(f"Error loading images at index {idx}: {e}")

        # Apply augmentations if specified
        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)

        # Convert to tensors if not already
        if not isinstance(image, Tensor):
            image = F.to_tensor(image)
        if not isinstance(mask, Tensor):
            mask = F.to_tensor(mask)
        return image, mask
    
    def __len__(self) -> int:
        available_images = len(self.paths['image'])
        if self.num_images > 0:
            return min(self.num_images, available_images)
        return available_images


class SegmentationDataset():
    """Class for managing train/val/test splits for segmentation datasets."""
    
    default_split: Dict[str, float] = {
        'train': 0.8,
        'val': 0.1,
        'test': 0.1
    }

    def __init__(
            self,
            dataset_loader: SegmentationDataLoader,
            dataset_split: Optional[Dict[str, float]] = None,
            batch_size: int = 8,
            shuffle: bool = True,
            num_workers: int = 4,
            pin_memory: bool = True
    ) -> None:
        """
        Initialize the segmentation dataset manager.
        
        Args:
            dataset_loader: SegmentationDataLoader instance
            dataset_split: Dictionary with split ratios (train, val, test)
            batch_size: Batch size for dataloaders
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
        """

        # Attributes
        self.dataset_loader = dataset_loader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Initialize dataset split with defaults if not provided
        if dataset_split is None:
            self.dataset_split = self.default_split.copy()
        else:
            self.dataset_split = dataset_split.copy()

            for key, value in self.default_split.items():
                if key not in self.dataset_split:
                    self.dataset_split[key] = value
                    
        # Normalize splits to sum to 1.0
        total = sum(self.dataset_split.values())

        if abs(total - 1.0) > 0.001:
            for key in self.dataset_split:
                self.dataset_split[key] /= total

    def get_split_lengths(self) -> Tuple[int, int, int]:
        """Calculate lengths for train/val/test splits."""

        dataset_len = len(self.dataset_loader)
        train = int(self.dataset_split['train']*dataset_len)
        val = int(self.dataset_split['val']*dataset_len)
        test = dataset_len - train - val
        return train, val, test
    
    def get_dataloaders(self, seed: int = 42) -> Dict[str, DataLoader]:
        """
        Create train/val/test dataloaders.
        
        Args:
            seed: Random seed for reproducible splits
        """

        # Generate random seed
        seed = torch.Generator().manual_seed(seed)
        # Get split lengths
        train_len, val_len, test_len = self.get_split_lengths()
        # Split the dataset
        splits = random_split(self.dataset_loader, [train_len, val_len, test_len], seed)

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

        # Log dataset split
        logging.info(f"Dataset split: train={train_len}, val={val_len}, test={test_len} samples")
        return loaders

