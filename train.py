import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import device, Tensor
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from score import iou_coef
import csv
from utils import make_folder
import time


class Train():
    def __init__(
            self,
            model: nn.Module,
            optimizer: optim,
            train_set: DataLoader,
            valid_set: DataLoader,
            test_set: DataLoader,
            epochs: int,
            accumulation_step: int,
            checkpoint_step: int,
            device: device,
            show: float,
            csv_path: str,
            checkpoint_path: str,
            model_path: str
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.epochs = epochs
        self.accumulation_step = accumulation_step
        self.checkpoint_step = checkpoint_step
        self.device = device
        self.show = show
        self.csv_path = csv_path
        self.checkpoint_path = checkpoint_path
        self.model_path = model_path

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
        self.scaler = GradScaler()
        self.criterion = nn.BCEWithLogitsLoss().to(device)

    def tensor_to_numpy(self, tensor: Tensor) -> np.float:
        return tensor.detach().cpu().numpy().astype(np.float32)
    
    def show_image(self, outputs: Tensor, masks: Tensor) -> None:
        output = torch.sigmoid(outputs[0][0])
        mask = masks[0][0]
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

    def save(self, save_path: str, save_name: str) -> None:
        make_folder(save_path)
        torch.save(self.model.state_dict(), save_name)
        print(f'Model saved at {save_name}.')

    def eval(self, dataset: DataLoader) -> float:
        num_dataset = len(dataset)
        if num_dataset > 0:
            loss, miou = 0, 0

            with torch.no_grad():
                self.model.eval()
                
                with autocast():
                    for inputs, masks in tqdm(dataset):
                        inputs, masks = inputs.to(self.device), masks.to(self.device)

                        # Forward propagation
                        outputs = self.model(inputs)

                        # Calculate loss
                        loss += self.criterion(outputs, masks).item()
                        miou += iou_coef(outputs, masks).item()

                        # Visualization
                        self.show_image(outputs, masks)

            loss /= num_dataset
            miou /= num_dataset
            return loss, miou
        else: raise ValueError('The length of dataset must be at least 1.')

    def train(self):
        # Record writer
        make_folder(self.csv_path)

        with open(f'{self.csv_path}train_logg.csv', 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, ['Epoch', 'Train_loss', 'Train_loss_mini', 'Train_loss_micro', 'Train_miou', 'Val_loss', 'Val_miou'])
            writer.writeheader()

            num_dataset = len(self.train_set)
            if num_dataset > 0:
                start_time = time.time()

                for epoch in range(1, self.epochs + 1):
                    self.model.train()
                    train_loss, train_loss_mini, train_loss_micro, train_miou = 0, 0, 0, 0

                    # Train
                    for i, (inputs, masks) in enumerate(tqdm(self.train_set), start=1):
                        inputs, masks = inputs.to(self.device), masks.to(self.device)

                        # Mixed precision learning: FP32 -> FP16
                        with autocast():
                            # Forward propagation
                            outputs = self.model(inputs)

                            # Calculate loss
                            loss = self.criterion(outputs, masks)
                            item = loss.item()

                            train_loss += item
                            train_loss_mini += item
                            train_loss_micro = item
                            train_miou += iou_coef(outputs, masks).item()

                        # Back propagation
                        self.scaler.scale(loss).backward()

                        # Gradient accumulation
                        if i % self.accumulation_step == 0:
                            # Unscale: FP16 -> FP32
                            self.scaler.unscale_(self.optimizer)

                            # Prevent gradient exploding
                            nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                            # Gradient update
                            self.scaler.step(self.optimizer)
                            self.scaler.update()

                            # Initialize gradient to zero
                            self.optimizer.zero_grad()
                            train_loss_mini /= self.accumulation_step
                            
                    train_loss /= num_dataset
                    train_miou /= num_dataset
                    print(f'Epoch: {epoch}, Train_loss: {train_loss}, Train_loss_mini: {train_loss_mini}, Train_loss_micro: {train_loss_micro}, Train_miou: {train_miou}')

                    # Evaluation
                    val_loss, val_miou = self.eval(self.valid_set)
                    self.scheduler.step(val_miou)
                    print(f'Epoch: {epoch}, Val_loss: {val_loss}, Val_miou: {val_miou}')

                    # Checkpoint save
                    if epoch % self.checkpoint_step == 0:
                        self.save(self.checkpoint_path, f'{self.checkpoint_path}epoch_{epoch}.pth')

                    # Record
                    writer.writerow({
                        'Epoch': epoch,
                        'Train_loss': train_loss,
                        'Train_loss_mini': train_loss_mini,
                        'Train_loss_micro': train_loss_micro,
                        'Train_miou': train_miou,
                        'Val_loss': val_loss,
                        'Val_miou': val_miou
                    })

                # Test
                test_loss, test_miou = self.eval(self.test_set)
                print(f'Test_loss: {test_loss}, Test_miou: {test_miou}')

                # Model save
                self.save(self.model_path, f'{self.model_path}model.pth')

                # Total train time
                end_time = time.time()
                total_time = end_time - start_time
                print(f'Total Train time: {round(total_time, 3)}s, {round(total_time/60, 3)}min, {round((total_time/60)/60, 3)}h.')
            else: raise ValueError('The length of dataset must be at least 1.')

