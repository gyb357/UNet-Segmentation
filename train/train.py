import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import time
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from torch import device, Tensor
from torch.amp import GradScaler, autocast
from loss import Loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# Columns for trainer configs
TRAINER_CONFIGS = [
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
            mixed_precision: bool = False,

            epochs: int = 100,
            accumulation_step: int = 1,
            checkpoint_step: int = 10,
            early_stopping_patience: int = 10,
    ) -> None:
        # Attributes
        self.device = device
        self.model = model.to(device)
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        self.epochs = epochs
        self.accumulation_step = accumulation_step
        self.checkpoint_step = checkpoint_step
        self.early_stopping_patience = early_stopping_patience

        # Loss functions
        self.criterion = loss_fn
        self.metrics_fn = metrics_fn
        self.metrics = Loss(num_classes=num_classes)

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
        self.model_path = ''
        self.log_path = ''
        self.checkpoint_path = ''
        self.best_model_path = ''


    def _mean_over_classes(
        self,
        fn: callable,
        preds: Tensor,
        masks: Tensor,
        method: str
    ) -> Tensor:
        return sum(
            fn(preds[:, c:c+1], (masks == c).float(), method)
            for c in range(self.num_classes)
        ) / self.num_classes
    
    def _get_activation(self, preds: Tensor) -> Tensor:
        if self.num_classes == 1:
            return torch.sigmoid(preds)
        else:
            return torch.softmax(preds, dim=1)
    
    def _save_model(self, epoch: int) -> None:
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        torch.save(self.model.state_dict(), os.path.join(self.model_path, f'epoch_{epoch}.pth'))
        print(f'Model saved at {os.path.join(self.model_path, f"epoch_{epoch}.pth")}')

    def eval(self, dataloader: DataLoader) -> Tuple[float, float]:
        # Check for empty dataset
        dataset_len = len(dataloader)
        if dataset_len == 0:
            raise ValueError('The length of dataset must be at least 1.')

        # Set model to evaluation mode
        self.model.eval()
        # Initialize metrics
        total_loss, total_metrics = 0, 0

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
                        metrics_loss = self.metrics.get_loss(preds, masks, self.metrics_fn)
                        metrics = self.metrics.get_coefficient(preds, masks, self.metrics_fn)
                    else:
                        metrics_loss = self._mean_over_classes(self.metrics.get_loss, preds, masks, self.metrics_fn)
                        metrics = self._mean_over_classes(self.metrics.get_coefficient, preds, masks, self.metrics_fn)

                # Integrate loss and metrics
                total_loss += (loss + metrics_loss).item()
                total_metrics += metrics.item()

            # Calculate mean loss and metrics
            total_loss /= dataset_len
            total_metrics /= dataset_len
            return total_loss, total_metrics
        
    def fit(self) -> None:
        # Check for empty dataset
        dataset_len = len(self.train_loader)
        if dataset_len == 0:
            raise ValueError('The length of training dataset must be at least 1.')
        
        # Define log recoder
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        with open(os.path.join(self.log_path, 'train.csv'), 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, TRAINER_CONFIGS)
            writer.writeheader()

            # Empty cache
            torch.cuda.empty_cache()

            for epoch in range(1, self.epochs + 1):
                # Set model to training mode
                self.model.train()
                # Initialize metrics
                train_loss, train_loss_mini, train_loss_micro, train_metrics = 0, 0, 0, 0

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
                            metrics_loss = self.metrics.get_loss(preds, masks, self.metrics_fn)
                            metrics = self.metrics.get_coefficient(preds, masks, self.metrics_fn)
                        else:
                            metrics_loss = self._mean_over_classes(self.metrics.get_loss, preds, masks, self.metrics_fn)
                            metrics = self._mean_over_classes(self.metrics.get_coefficient, preds, masks, self.metrics_fn)

                    # Integrate loss and metrics
                    total_loss = loss + metrics_loss
                    loss_item = total_loss.detach().item()
                    train_loss += loss_item
                    train_loss_mini += loss_item
                    train_loss_micro = loss_item
                    train_metrics += metrics

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
                writer.writerow({TRAINER_CONFIGS[i]: values[i] for i in range(len(TRAINER_CONFIGS))})
                csv_file.flush()

                # Log to tensorboard
                for i in range(1, len(TRAINER_CONFIGS)):
                    self.writer.add_scalar(TRAINER_CONFIGS[i], values[i], epoch)

                # Add sample images to tensorboard
                if epoch % 5 == 0 and self.valid_loader:
                    inputs, masks = next(iter(self.valid_loader))
                    inputs, masks = inputs.to(self.device), masks.to(self.device)

                    with torch.no_grad():
                        preds = self.model(inputs)
                        preds = self._get_activation(preds)

                    # Add images to tensorboard
                    self.writer.add_images('Input', inputs, epoch)
                    self.writer.add_images('Ground Truth', masks.unsqueeze(1), epoch)
                    self.writer.add_images('Prediction', preds, epoch)

                # Checkpoint
                if epoch % self.checkpoint_step == 0:
                    self._save_model(self.checkpoint_path, f'epoch_{epoch}.pth')

                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    print(f"Best validation metrics: {self.best_val_metrics:.4f} at epoch {best_epoch}")
                    break

            # Test
            test_loss, test_metrics = self.eval(self.test_loader)
            print(f"Test - Loss: {test_loss:.4f}, Metrics: {test_metrics:.4f}")
            print(f"Best validation metrics: {self.best_valid_metrics:.4f} at epoch {best_epoch}")

            # Save best model
            self._save_model(best_epoch)
            # Save last model
            self._save_model(epoch)

            # Close tensorboard
            self.tensorboard.close()

