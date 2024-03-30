import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch import Tensor
import numpy as np
import csv
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


class Train():
    def __init__(
            self,
            model: nn.Module,
            criterion,
            optim: optim,
            train_set,
            valid_set,
            test_set,
            epochs,
            accumulation,
            device,
            show,
            csv_path,
            model_path
    ):
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
        self.model_path = model_path

        self.scaler = GradScaler()

    def tensor_to_numpy(self, tensor: Tensor):
        return tensor.detach().cpu().numpy().astype(np.float32)

    def fit(self):
        with open(self.csv_path, 'w', newline='') as csv_file:
            columns = ['Epoch', 'Train_loss', 'Train_iou', 'Valid_loss', 'Valid_iou']
            writer = csv.DictWriter(csv_file, columns)
            writer.writeheader()
    
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                train_loss_epoch = 0
                train_loss_mini_batch = 0
                train_loss_micro_batch = 0
    
                for i, (inputs, masks) in enumerate(tqdm(self.train_set), start=1):
                    inputs, masks = inputs.to(self.device), masks.to(self.device)
    
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, masks)
                        train_loss_epoch += loss.item()
                        train_loss_mini_batch += loss.item()
                        train_loss_micro_batch = loss.item()
    
                    self.scaler.scale(loss).backward()
    
                    if i % self.accumulation == 0:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()
                        train_loss_mini_batch /= self.accumulation

                        if self.show > 0:
                            plt.figure(facecolor='black')
                            plt.subplot(2, 2, 1)
                            plt.imshow(self.tensor_to_numpy(outputs[0][0]))
                            plt.subplot(2, 2, 2)
                            plt.imshow(self.tensor_to_numpy(masks[0][0]))
                            plt.show(block=False)
                            plt.pause(self.show)
                            plt.close()
    
                train_loss_epoch /= len(self.train_set)
                print(f'train_loss_epoch: {train_loss_epoch}, train_loss_micro_batch: {train_loss_micro_batch}, train_loss_mini_batch: {train_loss_mini_batch}')

                with torch.no_grad():
                    self.model.eval()
                    val_loss = 0
                    val_miou = 0

                    for inputs, masks in tqdm(self.valid_set):
                        inputs, masks = inputs.to(self.device), masks.to(self.device)

                        outputs = self.model(inputs)
                        val_loss += self.criterion(outputs, masks).item()

                    val_loss /= len(self.valid_set)
                    print(f'Validation Epoch: {epoch}/{self.epochs}, Validation Loss: {val_loss}')

                writer.writerow({
                    'Epoch': epoch,
                    'Train_loss': train_loss_epoch,
                    'Train_iou': 0,
                    'Valid_loss': val_loss,
                    'Valid_iou': val_miou
                })

