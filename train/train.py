import torch.nn as nn
import torch.optim as optim
import os
import torch
import csv
import time
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from torch import device, Tensor
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from train.loss import dice_loss, dice_coefficient


# Columns for trainer configs
TRAINER_CONFIGS = [
    'Epoch',
    'Train_loss',
    'Train_loss_mini',
    'Train_loss_micro',
    'Train_metric',
    'Val_loss',
    'Val_metric',
    'Test_loss',
    'Test_metric',
    'Learning_rate'
]


class Trainer():
    """Trainer class for training the model"""
    
    def __init__(
            self,
            model: nn.Module,
            dataloader: Dict[str, DataLoader],
            lr: float,
            loss: nn.Module,
            loss_coefficient: Dict[str, float],
            device: device,
            epochs: int,
            accumulation_step: int,
            checkpoint_step: int,
            show_time: int,
            num_classes: int = 1,
            early_stopping_patience: int = 10,
            mixed_precision: bool = True,
            weight_decay: float = 0.0
    ) -> None:
        """
        Args:
            model (nn.Module): Model to be trained
            dataloader (Dict[str, DataLoader]): Dictionary containing train, val, and test dataloaders
            lr (float): Learning rate for the optimizer
            loss (nn.Module): Loss function
            loss_coefficient (Dict[str, float]): Coefficients for loss and metric
            device (device): Device to run the training on
            epochs (int): Number of epochs to train the model
            accumulation_step (int): Number of steps to accumulate gradients
            checkpoint_step (int): Number of epochs to save model checkpoint
            show_time (int): Number of steps to show sample images
            num_classes (int): Number of classes for the dataset
            early_stopping_patience (int): Number of epochs to wait before early stopping
            mixed_precision (bool): Whether to use mixed precision training
            weight_decay (float): Weight decay for the optimizer
        """

        # Attributes
        self.model = model
        self.dataloader = dataloader
        self.lr = lr
        self.device = device
        self.epochs = epochs
        self.accumulation_step = accumulation_step
        self.checkpoint_step = checkpoint_step
        self.show_time = show_time
        self.num_classes = num_classes
        self.early_stopping_patience = early_stopping_patience
        self.mixed_precision = mixed_precision
        self.weight_decay = weight_decay

        # Dataset
        self.train_loader = dataloader['train']
        self.val_loader = dataloader['val']
        self.test_loader = dataloader['test']

        # Train modules
        self.optim = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = loss
        self.scaler = GradScaler(enabled=mixed_precision)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optim,
            mode='max',
            patience=5,
            factor=0.5,
            verbose=True
        )

        # Loss coefficients
        self.loss_coefficient = loss_coefficient['loss']
        self.metric_coefficient = loss_coefficient['metric']

        # Tensorboard
        self.writer = SummaryWriter()

        # Early stopping variables
        self.best_val_metric = 0
        self.patience_counter = 0

        # Set for tracking best model
        self.best_model_state = None

    def _save_weights(self, path: str, name: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, name))
        print(f'Model saved at {os.path.join(path, name)}')

    def _get_activation(self, outputs: Tensor) -> Tensor:
        if self.num_classes == 1:
            return torch.sigmoid(outputs)
        else:
            return torch.softmax(outputs, dim=1)
        
    def evaluate(self, dataloader: DataLoader, desc: str = "Evaluating") -> Tuple[float, float]:
        dataset_len = len(dataloader)

        if dataset_len == 0:
            raise ValueError('The length of dataset must be at least 1.')

        self.model.eval()
        total_loss, total_metrics = 0, 0
        
        with torch.no_grad():
            for inputs, masks in tqdm(dataloader, desc=desc):
                inputs, masks = inputs.to(self.device), masks.to(self.device)

                # Mixed precision inference
                with autocast(device_type=self.device.type, enabled=self.mixed_precision):
                    # Forward propagation
                    outputs = self.model(inputs)

                    # Calculate loss
                    loss = self.criterion(outputs, masks) * self.loss_coef
                    # Use appropriate loss based on number of classes
                    if self.num_classes == 1:
                        metrics_loss = dice_loss(outputs, masks, 1) * self.metrics_coef
                        metrics = dice_coefficient(outputs, masks, 1)
                    else:
                        # For multi-class, calculate loss over all classes
                        metrics_loss = sum(dice_loss(outputs[:, c:c+1], 
                                                   (masks == c).float(), 1) 
                                         for c in range(self.num_classes)) / self.num_classes * self.metrics_coef
                        metrics = sum(dice_coefficient(outputs[:, c:c+1], 
                                                     (masks == c).float(), 1) 
                                    for c in range(self.num_classes)) / self.num_classes

                total_loss += (loss + metrics_loss).item()
                total_metrics += metrics

                # Visualization
                # show_image(self.show_time, outputs, masks, self.num_classes)

        total_loss /= dataset_len
        total_metrics /= dataset_len
        return total_loss, total_metrics
    
    def fit(self, csv_path: str, csv_name: str, checkpoint_path: str, model_path: str) -> None:
        dataset_len = len(self.train_loader)

        if dataset_len == 0:
            raise ValueError('The length of training dataset must be at least 1.')
        
        # Define log recorder
        os.makedirs(csv_path, exist_ok=True)
        
        # Initialize tracking variables
        best_epoch = 0

        with open(os.path.join(csv_path, csv_name), 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, TRAINER_CONFIGS)
            writer.writeheader()

            # Train start time
            start_time = time.time()

            # empty the cache
            torch.cuda.empty_cache()

            # Train
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                train_loss, train_loss_mini, train_loss_micro, train_metrics = 0, 0, 0, 0

                # Create progress bar for training loop
                progress_bar = tqdm(enumerate(self.train_loader, start=1), total=len(self.train_loader), desc=f"Epoch {epoch}/{self.epochs}")
                
                for i, (inputs, masks) in progress_bar:
                    inputs, masks = inputs.to(self.device), masks.to(self.device)
    
                    # Mixed precision learning: FP32 -> FP16
                    with autocast(device_type=self.device.type, enabled=self.mixed_precision):
                        # Forward propagation
                        outputs = self.model(inputs)
    
                        # Calculate loss
                        loss = self.criterion(outputs, masks) * self.loss_coefficient
                        # Use appropriate loss based on number of classes
                        if self.num_classes == 1:
                            metrics_loss = dice_loss(outputs, masks, 1)*self.metric_coefficient
                            metrics = dice_coefficient(outputs, masks, 1)
                        else:
                            # For multi-class, calculate loss over all classes
                            metrics_loss = sum(
                                dice_loss(
                                    outputs[:, c:c+1], 
                                    (masks == c).float(), 1
                                ) 
                                for c in range(self.num_classes)
                            )/self.num_classes*self.metric_coefficient
                            metrics = sum(
                                dice_coefficient(
                                    outputs[:, c:c+1], 
                                    (masks == c).float(), 1
                                ) 
                                for c in range(self.num_classes)
                            )/self.num_classes
                        
                        total_loss = loss + metrics_loss
    
                    # Track metrics
                    loss_item = total_loss.item()
                    train_loss += loss_item
                    train_loss_mini += loss_item
                    train_loss_micro = loss_item
                    train_metrics += metrics
    
                    # Back propagation
                    self.scaler.scale(total_loss).backward()
    
                    # Gradient accumulation
                    if i % self.accumulation_step == 0:
                        # Unscale: FP16 -> FP32
                        self.scaler.unscale_(self.optim)
    
                        # Prevent gradient exploding
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1)
    
                        # Gradient update
                        self.scaler.step(self.optim)
                        self.scaler.update()
    
                        # Initialize gradient to zero
                        self.optim.zero_grad(set_to_none=True)  # More efficient than zero_grad()
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
                val_loss, val_metrics = self.evaluate(self.val_loader, desc="Validating")
                test_loss, test_metrics = self.evaluate(self.test_loader, desc="Testing")
                
                # Get current learning rate
                current_lr = self.optim.param_groups[0]['lr']
                
                # Update learning rate scheduler
                self.scheduler.step(val_metrics)
                
                # Early stopping check
                if val_metrics > self.best_val_metrics:
                    self.best_val_metrics = val_metrics
                    self.patience_counter = 0
                    self.best_model_state = self.model.state_dict().copy()
                    best_epoch = epoch
                    # Save best model
                    self._save_weights(model_path, 'best_model.pth')
                else:
                    self.patience_counter += 1
                
                # Print epoch results
                print(f'Epoch: {epoch}/{self.epochs}')
                print(f'Train - Loss: {train_loss:.4f}, Metrics: {train_metrics:.4f}')
                print(f'Val   - Loss: {val_loss:.4f}, Metrics: {val_metrics:.4f}')
                print(f'Test  - Loss: {test_loss:.4f}, Metrics: {test_metrics:.4f}')
                print(f'LR: {current_lr:.6f}, Best val metrics: {self.best_val_metrics:.4f} (epoch {best_epoch})')
                
                # Record training log
                values = [
                    epoch, train_loss, train_loss_mini, train_loss_micro, 
                    train_metrics, val_loss, val_metrics, test_loss, test_metrics, current_lr
                ]
                data = {TRAINER_CONFIGS[i]: values[i] for i in range(len(TRAINER_CONFIGS))}
                writer.writerow(data)
                csv_file.flush() # Ensure data is written immediately

                # Log to tensorboard
                for i in range(1, len(TRAINER_CONFIGS)):
                    self.writer.add_scalar(TRAINER_CONFIGS[i], values[i], epoch)
                
                # Add sample images to tensorboard
                if epoch % 5 == 0 and self.val_set:
                    inputs, masks = next(iter(self.val_set))
                    inputs = inputs.to(self.device)
                    masks = masks.to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        outputs = self._get_activation(outputs)
                    
                    # Add images to tensorboard
                    self.writer.add_images('Input', inputs, epoch)
                    self.writer.add_images('Ground Truth', masks.unsqueeze(1), epoch)
                    self.writer.add_images('Prediction', outputs, epoch)

                # Checkpoint save
                if epoch % self.checkpoint_step == 0:
                    self._save_weights(checkpoint_path, f'epoch_{epoch}.pth')
                
                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    print(f'Early stopping triggered after {epoch} epochs.')
                    print(f'Best validation metrics: {self.best_val_metrics:.4f} at epoch {best_epoch}')
                    break
            
            # Restore best model
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
                print(f'Restored best model from epoch {best_epoch}')
                    
            # Final model save
            self._save_weights(model_path, 'final_model.pth')
    
            # Total train time
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f'Training completed in: {int(hours)}h {int(minutes)}m {seconds:.2f}s')
            print(f'Best validation metrics: {self.best_val_metrics:.4f} at epoch {best_epoch}')
    
            self.writer.close()

