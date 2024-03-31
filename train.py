import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import device, Tensor
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from loss import dice_loss
import torch.nn.functional as F
import torch


class Train():
    def __init__(
            self,
            model: nn.Module,
            criterion: nn,
            optim: optim,
            train_set: DataLoader,
            valid_set: DataLoader,
            test_set: DataLoader,
            epochs: int,
            accumulation: int,
            device: device,
            show: float,
            csv_path: str,
            checkpoint_path: str,
            model_path: str
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optim = optim
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.epochs = epochs
        self.accumulation = accumulation
        self.device = device
        self.show = show
        self.csv_path = csv_path
        self.checkpoint_path = checkpoint_path
        self.model_path = model_path
        self.scaler = GradScaler()

    def tensor_to_numpy(self, tensor: Tensor) -> np.float:
        return tensor.detach().cpu().numpy().astype(np.float32)
    
    def show_image(self, outputs: Tensor, masks: Tensor) -> None:
        output, mask = outputs[0][0], masks[0][0]
        if self.show > 0:
            plt.subplot(1, 2, 1)
            plt.title('Predicted mask')
            plt.imshow(self.tensor_to_numpy(output))

            plt.subplot(1, 2, 2)
            plt.title('Ground truth')
            plt.imshow(self.tensor_to_numpy(mask))

            plt.show(block=False)
            plt.pause(self.show)
            plt.close()

    def test(self, dataset: DataLoader):
        num_dataset = len(dataset)
        
        if num_dataset > 0:
            loss = 0
            dice = 0

            with torch.no_grad():
                for inputs, masks in tqdm(dataset):
                    inputs, masks = inputs.to(self.device), masks.to(self.device)

                    outputs = self.model(inputs)
                    loss += self.criterion(outputs, masks).item()
                    dice += dice_loss(outputs, masks)
                    self.show_image(outputs, masks)

                loss /= num_dataset
                dice /= num_dataset
            return loss, dice
        else:
            raise ValueError('The length of dataset must be at least 1.')
        
    def fit(self):
        with open(self.csv_path, 'w', newline='') as csv_file:
            # Recorders
            columns = ['Epoch', 'Train_loss', 'Train_dice_loss', 'Val_loss', 'Val_dice_loss']
            writer = csv.DictWriter(csv_file, columns)
            writer.writeheader()

            for epoch in range(1, self.epochs + 1):
                self.model.train()
                train_loss = 0
                train_loss_mini = 0
                train_loss_micro = 0

                val_loss = 0
                val_dice_loss = 0


                # Train
                if len(self.train_set) > 0:
                    for i, (inputs, masks) in enumerate(tqdm(self.train_set), start=1):
                        inputs, masks = inputs.to(self.device), masks.to(self.device)

                        # Mixed precision learning: FP32 -> FP16
                        with autocast():
                            # Forward propagation
                            outputs = self.model(inputs)
                            # Calculate loss
                            loss = self.criterion(outputs, masks)
                            dice = dice_loss(outputs, masks)
                            loss += dice

                            train_loss += loss.item()
                            train_loss_mini += loss.item()
                            train_loss_micro = loss.item()

                        # Back propagation
                        self.scaler.scale(loss).backward()

                        # Gradient accumulation
                        if i % self.accumulation == 0:
                            # Unscale: FP16 -> FP32
                            self.scaler.unscale_(self.optim)
                            # Prevent gradient runaway
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                            # Gradient update
                            self.scaler.step(self.optim)
                            self.scaler.update()
                            # Initialize gradient to zero
                            self.optim.zero_grad()
                            train_loss_mini /= self.accumulation

                    # Visualization
                    self.show_image(outputs, masks)
                    train_loss /= len(self.train_set)
                    print(f'Epoch: {epoch}, train_loss: {train_loss}, train_loss_mini: {train_loss_mini}, train_loss_micro: {train_loss_micro}, train_dice_loss: {dice}')
                else:
                    raise ValueError('The length of train_set must be at least 1.')
                

                # Valid
                val_loss, val_dice_loss = self.test(self.valid_set)
                print(f'Epoch: {epoch}, val_loss: {val_loss}, val_dice_loss: {val_dice_loss}')

                # Record
                writer.writerow({
                    'Epoch': epoch,
                    'Train_loss': train_loss,
                    'Train_dice_loss': dice,
                    'Val_loss': val_loss,
                    'Val_dice_loss': val_dice_loss
                })

