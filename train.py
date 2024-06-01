from typing import List, Tuple, Dict
import os
import csv
from torch import Tensor
import matplotlib.pyplot as plt
from utils import tensor_to_numpy
import pandas as pd
from unet import UNet
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from time import time
from tqdm import tqdm
from miou import miou_coef


csv_name = 'train_logg.csv'
columns = [
    'Epoch',
    'Train_loss',
    'Train_loss_mini',
    'Train_loss_micro',
    'Train_miou',
    'Val_loss',
    'Val_miou'
]
csv_path = 'csv/'
checkpoint_path = 'model/checkpoint/'
model_path = 'model/'


def get_csv_DictWriter(path: str, name: str, column: List[str]) -> Tuple[csv.DictWriter, any]:
    os.makedirs(path, exist_ok=True)
    csv_file = open(os.path.join(path, name), mode='w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=column)
    writer.writeheader()
    return writer, csv_file


def show_image(show_time: float, output: Tensor, mask: Tensor) -> None:
    if show_time > 0:
        plt.subplot(1, 2, 1)
        plt.title('Predicted mask')
        plt.imshow(tensor_to_numpy(output))

        plt.subplot(1, 2, 2)
        plt.title('Ground truth')
        plt.imshow(tensor_to_numpy(mask))
        
        plt.show(block=False)
        plt.pause(show_time)
        plt.close()


def show_plot() -> None:
    data = pd.read_csv(os.path.join(csv_path, csv_name))
    x = data[columns[0]]

    plt.figure()
    for col in columns[1:]:
        plt.plot(x, data[col], label=col)

    plt.xlabel(columns[0])
    plt.legend()
    plt.show()


class Trainer():
    def __init__(
            self,
            model: UNet,
            dataset: Dict[str, DataLoader],
            lr: float,
            device: torch.device,
            epochs: int,
            accumulation_step: int,
            checkpoint_step: int,
            show_time: float,
            show_plt: bool
    ) -> None:
        required_keys = ['train', 'val', 'test']
        if not all(key in dataset for key in required_keys):
            missing_keys = [key for key in required_keys if key not in dataset]
            raise ValueError(f"Dataset is missing the following keys: {', '.join(missing_keys)}")

        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.device = device
        self.epochs = epochs
        self.accumulation_step = accumulation_step
        self.checkpoint_step = checkpoint_step
        self.show_time = show_time
        self.show_plt = show_plt

        self.train_set = dataset['train']
        self.val_set = dataset['val']
        self.test_set = dataset['test']

        self.optim = optim.Adam(model.parameters(), lr)
        self.criterion = nn.BCEWithLogitsLoss().to(device)
        self.scaler = GradScaler()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'max', patience=5)

        self.csv_writer, self.csv_file = get_csv_DictWriter(csv_path, csv_name, columns)
        self.tensorboard = SummaryWriter()

    def __del__(self):
        self.csv_file.close()

    def save_model(self, save_path: str, save_name: str) -> None:
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, save_name))
        print(f'Model saved at {save_name}.')

    def eval(self, dataset: DataLoader) -> Tuple[float, float]:
        dataset_len = len(dataset)
        if dataset_len == 0:
            raise ValueError('The length of dataset must be at least 1.')

        loss, miou = 0, 0
        self.model.eval()
        with torch.no_grad():
            for inputs, masks in tqdm(dataset):
                inputs, masks = inputs.to(self.device), masks.to(self.device)
                with autocast():
                    outputs = self.model(inputs)
                    loss += self.criterion(outputs, masks).item()
                    miou += miou_coef(outputs, masks).item()
                    show_image(self.show_time, outputs, masks)

        loss /= dataset_len
        miou /= dataset_len
        return loss, miou
    
    def train(self) -> None:
        if len(self.train_set) == 0:
            raise ValueError('The length of training dataset must be at least 1.')

        start_time = time()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss, train_loss_mini, train_loss_micro, train_miou = 0, 0, 0, 0

            for i, (inputs, masks) in enumerate(tqdm(self.train_set), start=1):
                inputs, masks = inputs.to(self.device), masks.to(self.device)
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, masks)
                    loss_item = loss.item()

                    train_loss += loss_item
                    train_loss_mini += loss_item
                    train_loss_micro = loss_item
                    train_miou += miou_coef(outputs, masks).item()

                self.scaler.scale(loss).backward()

                if i % self.accumulation_step == 0:
                    self.scaler.unscale_(self.optim)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    self.optim.zero_grad()
                    train_loss_mini /= self.accumulation_step

            train_loss /= len(self.train_set)
            train_miou /= len(self.train_set)
            print(f'Epoch: {epoch}, Train_loss: {train_loss}, Train_loss_mini: {train_loss_mini}, Train_loss_micro: {train_loss_micro}, Train_miou: {train_miou}')

            val_loss, val_miou = self.eval(self.val_set)
            self.scheduler.step(val_miou)
            print(f'Epoch: {epoch}, Val_loss: {val_loss}, Val_miou: {val_miou}')

            test_loss, test_miou = self.eval(self.test_set)
            print(f'Test_loss: {test_loss}, Test_miou: {test_miou}')

            csv_row = dict(zip(columns, [epoch, train_loss, train_loss_mini, train_loss_micro, train_miou, val_loss, val_miou]))
            self.csv_writer.writerow(csv_row)

            for col, val in zip(columns[1:], [train_loss, train_loss_mini, train_loss_micro, train_miou, val_loss, val_miou]):
                self.tensorboard.add_scalar(col, val, epoch)

            if epoch % self.checkpoint_step == 0:
                self.save_model(checkpoint_path, f'epoch_{epoch}.pth')

        elapsed_time = time() - start_time
        print(f'Training completed in: {elapsed_time:.2f} seconds')

