from typing import Tuple, Dict
import csv
import os
from torch import Tensor
import matplotlib.pyplot as plt
from utils import tensor_to_numpy
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import device
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torch
from miou import miou_coef
from tqdm import tqdm
import time


COLUMNS = [
    'Epoch',
    'Train_loss',
    'Train_loss_mini',
    'Train_loss_micro',
    'Train_miou',
    'Val_loss',
    'Val_miou'
]


def show_image(show_time: float, output: Tensor, mask: Tensor) -> None:
    if show_time > 0:
        output = torch.sigmoid(output)

        plt.subplot(1, 2, 1)
        plt.title('Predicted mask')
        plt.imshow(tensor_to_numpy(output[0][0]))

        plt.subplot(1, 2, 2)
        plt.title('Ground truth')
        plt.imshow(tensor_to_numpy(mask[0][0]))
        
        plt.show(block=False)
        plt.pause(show_time)
        plt.close()


class Trainer():
    def __init__(
            self,
            model: nn.Module,
            dataset: Dict[str, DataLoader],
            lr: float,
            device: device,
            epochs: int,
            accumulation_step: int,
            checkpoint_step: int,
            show_time: int
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.device = device
        self.epochs = epochs
        self.accumulation_step = accumulation_step
        self.checkpoint_step = checkpoint_step
        self.show_time = show_time

        # Dataset
        self.train_set = dataset['train']
        self.val_set = dataset['val']
        self.test_set = dataset['test']

        # Train modules
        self.optim = optim.Adam(model.parameters(), lr)
        self.criterion = nn.BCEWithLogitsLoss().to(device)
        self.scaler = GradScaler()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'max', patience=5)

        # Tensorboard
        self.tensorboard = SummaryWriter()

    def save_model(self, path: str, name: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, name))
        print(f'Model saved at {name}.')

    def evaluate(self, dataset: DataLoader) -> Tuple[float, float]:
        dataset_len = len(dataset)
        if dataset_len == 0:
            raise ValueError('The length of dataset must be at least 1.')

        total_loss, total_miou = 0, 0
        self.model.eval()
        with torch.no_grad():
            for inputs, masks in tqdm(dataset):
                inputs, masks = inputs.to(self.device), masks.to(self.device)
                with autocast():
                    # Forward propagation
                    outputs = self.model(inputs)

                    # Calculate loss
                    total_loss += self.criterion(outputs, masks).item()
                    total_miou += miou_coef(outputs, masks).item()

                    # Visualization
                    show_image(self.show_time, outputs, masks)

        avg_loss = total_loss/dataset_len
        avg_miou = total_miou/dataset_len
        return avg_loss, avg_miou
    
    def train(self, csv_path: str, csv_name: str, checkpoint_path: str, model_path: str) -> None:
        if len(self.train_set) == 0:
            raise ValueError('The length of training dataset must be at least 1.')
        
        # Define logg recoder
        os.makedirs(csv_path, exist_ok=True)
        with open(f'{csv_path}{csv_name}', 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, COLUMNS)
            writer.writeheader()

            # Train start time
            start_time = time.time()

            # Train
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                train_loss, train_loss_mini, train_loss_micro, train_miou = 0, 0, 0, 0

                for i, (inputs, masks) in enumerate(tqdm(self.train_set), start=1):
                    inputs, masks = inputs.to(self.device), masks.to(self.device)
    
                    # Mixed precision learning: FP32 -> FP16
                    with autocast():
                        # Forward propagation
                        outputs = self.model(inputs)
    
                        # Calculate loss
                        loss = self.criterion(outputs, masks)
                        loss_item = loss.item()
                        train_loss += loss_item
                        train_loss_mini += loss_item
                        train_loss_micro = loss_item
                        train_miou += miou_coef(outputs, masks).item()
    
                    # Back propagation
                    self.scaler.scale(loss).backward()
    
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
                        self.optim.zero_grad()
                        train_loss_mini /= self.accumulation_step
    
                train_loss /= len(self.train_set)
                train_miou /= len(self.train_set)
                print(f'Epoch: {epoch}, Train_loss: {train_loss}, Train_loss_mini: {train_loss_mini}, Train_loss_micro: {train_loss_micro}, Train_miou: {train_miou}')
    
                # Evaluation
                val_loss, val_miou = self.evaluate(self.val_set)
                self.scheduler.step(val_miou)
                print(f'Epoch: {epoch}, Val_loss: {val_loss}, Val_miou: {val_miou}')

                # Train logg recode
                values = [epoch, train_loss, train_loss_mini, train_loss_micro, train_miou, val_loss, val_miou]
                data = {COLUMNS[i]: values[i] for i in range(len(COLUMNS))}
                writer.writerow(data)

                for i in range(1, len(COLUMNS)):
                    self.tensorboard.add_scalar(COLUMNS[i], values[i], epoch)

                # Test
                test_loss, test_miou = self.evaluate(self.test_set)
                print(f'Test_loss: {test_loss}, Test_miou: {test_miou}')

                # Checkpoint save
                if epoch % self.checkpoint_step == 0:
                    self.save_model(checkpoint_path, f'epoch_{epoch}.pth')
                    
            # Model save
            self.save_model(model_path, 'model.pth')
    
            # Total train time
            elapsed_time = time.time() - start_time
            print(f'Training completed in: {elapsed_time:.2f} seconds')
    
            self.tensorboard.close()

