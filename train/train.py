import torch.nn as nn
import torch.optim as optim
import torch
import os
import csv
from torch import device, Tensor
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from train.loss import Loss
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# Columns for trainer configs
_TRAINER_CONFIGS = [
    'Epoch',
    'Train_loss',
    'Train_loss_mini',
    'Train_loss_micro',
    'Train_metrics',
    'Valid_loss',
    'Valid_metrics',
    'Learning_rate'
]


class Trainer():
    def __init__(
            self,
            device: device,
            model: nn.Module,
            loss_fn: nn.Module,
            metrics_fn: str,
            num_classes: int,
            dataloader: Dict[str, DataLoader],
            lr: float = 1e-4,
            weight_decay: float = 0.0,
            mixed_precision: bool = True,
            epochs: int = 100,
            accumulation_step: int = 1,
            checkpoint_step: int = 10,
            early_stopping_patience: int = 10,
    ) -> None:
        """
        Args:
            device (device): Device to run the model on ('cuda' or 'cpu')
            model (nn.Module): Model to train
            loss_fn (nn.Module): Loss function ('BCEWithLogitsLoss' or 'CrossEntropyLoss')
            metrics_fn (str): Metrics function ('dice' or 'iou')
            num_classes (int): Number of classes
            dataloader (dict): Data loaders for 'train', 'valid', 'test'
            lr (float): Learning rate (default: 1e-4)
            weight_decay (float): Weight decay (default: 0.0)
            mixed_precision (bool): Whether to use mixed precision (default: False)
            epochs (int): Number of epochs (default: 100)
            accumulation_step (int): Number of steps to accumulate gradients (default: 1)
            checkpoint_step (int): Number of epochs to save checkpoint (default: 10)
            early_stopping_patience (int): Number of epochs to wait before early stopping (default: 10)
        """

        # Attributes
        self.device = torch.device(device)
        self.model = model.to(device)
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        self.epochs = epochs
        self.accumulation_step = accumulation_step
        self.checkpoint_step = checkpoint_step
        self.early_stopping_patience = early_stopping_patience

        # Loss functions
        self.criterion = loss_fn
        self.metrics = Loss(num_classes, metrics_fn)

        # Dataset splits
        self.train_loader = dataloader['train']
        self.valid_loader = dataloader['valid']
        self.test_loader = dataloader['test']

        # Training modules
        self.optim = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = GradScaler(enabled=mixed_precision)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optim,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Early stopping
        self.best_valid_metrics = 0
        self.patience_counter = 0
        self.best_model_state = None

        # Tensorboard
        self.tensorboard = SummaryWriter()
        # Paths
        self.model_dir = './model'
        self.log_dir = './log'
        self.checkpoint_dir = './checkpoint'
        self.best_model_dir = './best_model'

    def _check_data_length(self, dataloader: DataLoader) -> int:
        """
        Args:
            dataloader (DataLoader): Data loader
        """

        dataset_len = len(dataloader)
        if dataset_len == 0:
            raise ValueError('The length of dataset must be at least 1.')
        return dataset_len

    def _mean_over_classes(self, fn: callable, preds: Tensor, masks: Tensor) -> Tensor:
        """
        Args:
            fn (callable): Function to apply to each class
            preds (Tensor): Predictions
            masks (Tensor): Masks
        """

        return sum(
            fn(preds[:, c:c+1], (masks == c).float())
            for c in range(self.num_classes)
        ) / self.num_classes
    
    def _get_activation(self, preds: Tensor) -> Tensor:
        """
        Args:
            preds (Tensor): Predictions
        """

        if self.num_classes == 1:
            return torch.sigmoid(preds)
        else:
            return torch.softmax(preds, dim=1)
    
    def _save_model(self, epoch: int, dir: str) -> None:
        """
        Args:
            epoch (int): Epoch number
            dir (str): Directory to save the model
        """

        if not os.path.exists(dir):
            os.makedirs(dir)

        # Save model
        torch.save(
            self.model.state_dict(),
            os.path.join(dir, f'epoch_{epoch}.pth')
        )
        print(f'Model saved at {os.path.join(dir, f"epoch_{epoch}.pth")}')

    def eval(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Args:
            dataloader (DataLoader): Data loader
        """

        # Check for empty dataset
        dataset_len = self._check_data_length(dataloader)
        # Set model to evaluation mode
        self.model.eval()
        # Initialize metrics
        total_loss, total_metrics = 0.0, 0.0

        with torch.no_grad():
            for inputs, masks in tqdm(dataloader, desc='Evaluating'):
                inputs, masks = inputs.to(self.device), masks.to(self.device)

                # Mixed precision inference
                with autocast(device_type=self.device.type, enabled=self.mixed_precision):
                    # Forward propagation
                    preds = self.model(inputs)

                    # Calculate loss
                    loss = self.criterion(preds, masks)

                    # Calculate metrics
                    if self.num_classes == 1:
                        metrics_loss = self.metrics.get_loss(preds, masks)
                        metrics = self.metrics.get_coefficient(preds, masks)
                    else:
                        metrics_loss = self._mean_over_classes(self.metrics.get_loss, preds, masks)
                        metrics = self._mean_over_classes(self.metrics.get_coefficient, preds, masks)

                # Integrate loss and metrics
                total_loss += (loss + metrics_loss).item()
                total_metrics += metrics.item()

            # Calculate mean loss and metrics
            total_loss /= dataset_len
            total_metrics /= dataset_len
            return total_loss, total_metrics
        
    def fit(self) -> None:
        # Check for empty dataset
        dataset_len = self._check_data_length(self.train_loader)
        
        # Define log recoder
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        with open(os.path.join(self.log_dir, 'train.csv'), 'w', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, _TRAINER_CONFIGS)
            csv_writer.writeheader()

            # Empty cache
            torch.cuda.empty_cache()

            for epoch in range(1, self.epochs + 1):
                # Set model to training mode
                self.model.train()
                # Initialize metrics
                train_loss, train_loss_mini, train_loss_micro, train_metrics = 0.0, 0.0, 0.0, 0.0

                # Create progress bar for training loop
                progress_bar = tqdm(enumerate(self.train_loader, start=1), total=dataset_len, desc=f"Epoch {epoch}/{self.epochs}")

                for i, (inputs, masks) in progress_bar:
                    inputs, masks = inputs.to(self.device), masks.to(self.device)

                    # Mixed precision learning: FP32 -> FP16
                    with autocast(device_type=self.device.type, enabled=self.mixed_precision):
                        # Forward propagation
                        preds = self.model(inputs)

                        # Calculate loss
                        loss = self.criterion(preds, masks)

                        # Calculate metrics
                        if self.num_classes == 1:
                            metrics_loss = self.metrics.get_loss(preds, masks)
                            metrics = self.metrics.get_coefficient(preds, masks)
                        else:
                            metrics_loss = self._mean_over_classes(self.metrics.get_loss, preds, masks)
                            metrics = self._mean_over_classes(self.metrics.get_coefficient, preds, masks)

                    # Integrate loss and metrics
                    total_loss = loss + metrics_loss
                    loss_item = total_loss.detach().item()
                    train_loss += loss_item
                    train_loss_mini += loss_item
                    train_loss_micro = loss_item
                    train_metrics += metrics.item()

                    # Scaler backward
                    self.scaler.scale(total_loss).backward()

                    # Gradient accumulation
                    if i % self.accumulation_step == 0:
                        # Unscale: FP16 -> FP32
                        self.scaler.unscale_(self.optim)
                        # Prevent gradient exploding
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        # Gradient update
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        # Initialize gradient to zero
                        self.optim.zero_grad(set_to_none=True)

                        # Calculate mini-batch loss
                        train_loss_mini /= self.accumulation_step

                        # Update progress bar with current metrics
                        progress_bar.set_postfix({
                            'loss': f'{train_loss_micro:.4f}',
                            'metrics': f'{metrics:.4f}',
                            'lr': f'{self.optim.param_groups[0]["lr"]:.6f}'
                        })

                # Calculate epoch metrics
                train_loss /= dataset_len
                train_metrics /= dataset_len

                # Evaluation
                valid_loss, valid_metrics = self.eval(self.valid_loader)
                
                # Get current learning rate
                current_lr = self.optim.param_groups[0]['lr']

                # Update learning rate scheduler
                self.scheduler.step(valid_metrics)
                
                # Update best model
                if valid_metrics > self.best_valid_metrics:
                    self.best_valid_metrics = valid_metrics
                    self.patience_counter = 0
                    self.best_model_state = self.model.state_dict()

                    best_epoch = epoch
                else:
                    self.patience_counter += 1

                # Print epoch results
                print(f"Epoch: {epoch}/{self.epochs}")
                print(f"Train - Loss: {train_loss:.4f}, Metrics: {train_metrics:.4f}")
                print(f"Valid - Loss: {valid_loss:.4f}, Metrics: {valid_metrics:.4f}")
                print(f"Best validation metrics: {self.best_valid_metrics:.4f} at epoch {best_epoch}")
                print(f"LR: {current_lr:.6f}, Best val metrics: {self.best_valid_metrics:.4f} (epoch {best_epoch})")

                # Record training log
                values = [
                    epoch,
                    train_loss,
                    train_loss_mini,
                    train_loss_micro,
                    train_metrics,
                    valid_loss,
                    valid_metrics,
                    current_lr
                ]

                # Log to csv
                csv_writer.writerow({_TRAINER_CONFIGS[i]: values[i] for i in range(len(_TRAINER_CONFIGS))})
                csv_file.flush()
                # Log to tensorboard
                for i in range(1, len(_TRAINER_CONFIGS)):
                    self.tensorboard.add_scalar(_TRAINER_CONFIGS[i], values[i], epoch)

                # Add sample images to tensorboard
                if epoch % 5 == 0 and self.valid_loader:
                    inputs, masks = next(iter(self.valid_loader))
                    inputs, masks = inputs.to(self.device), masks.to(self.device)

                    with torch.no_grad():
                        preds = self.model(inputs)
                        preds = self._get_activation(preds)

                    # Add images to tensorboard
                    self.tensorboard.add_images('Input', inputs, epoch)
                    self.tensorboard.add_images('Ground Truth', masks, epoch)
                    self.tensorboard.add_images('Prediction', preds, epoch)

                # Checkpoint
                if epoch % self.checkpoint_step == 0:
                    self._save_model(epoch, self.checkpoint_dir)

                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

            # Test
            if self.test_loader:
                test_loss, test_metrics = self.eval(self.test_loader)
                print(f"Test - Loss: {test_loss:.4f}, Metrics: {test_metrics:.4f}")

            # Save best model
            self._save_model(best_epoch, self.best_model_dir)
            # Save last model
            self._save_model(epoch, self.model_dir)

            # Close tensorboard
            self.tensorboard.close()
            print('Training completed.')

