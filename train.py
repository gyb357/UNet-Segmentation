import torch.nn as nn
from torch.utils.data import DataLoader
from torch import device, Tensor
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import csv
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from utils import make_folder, tensor_to_numpy
import torch
from tqdm import tqdm
from score import miou_coef, miou_loss
import time
import pandas as pd


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


class Train():
    def __init__(
            self,
            model: nn.Module,
            dataset: dict,
            lr: float,
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
        self.dataset = dataset
        self.lr = lr
        self.epochs = epochs
        self.accumulation_step = accumulation_step
        self.checkpoint_step = checkpoint_step
        self.device = device
        self.show = show
        self.csv_path = csv_path
        self.checkpoint_path = checkpoint_path
        self.model_path = model_path

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

    def save(self, save_path: str, save_name: str) -> None:
        make_folder(save_path)
        torch.save(self.model.state_dict(), save_name)
        print(f'Model saved at {save_name}.')

    def show_image(self, outputs: Tensor, masks: Tensor) -> None:
        output = torch.sigmoid(outputs[0][0])
        mask = masks[0][0]
        if self.show > 0:
            plt.subplot(1, 2, 1)
            plt.title('Predicted mask')
            plt.imshow(tensor_to_numpy(output))

            plt.subplot(1, 2, 2)
            plt.title('Ground truth')
            plt.imshow(tensor_to_numpy(mask))

            plt.show(block=False)
            plt.pause(self.show)
            plt.close()

    def eval(self, dataset: DataLoader):
        dataset_len = len(dataset)
        if dataset_len > 0:
            loss, miou = 0, 0

            with torch.no_grad():
                self.model.eval()
                
                for inputs, masks in tqdm(dataset):
                    inputs, masks = inputs.to(self.device), masks.to(self.device)

                    with autocast():
                        # Forward propagation
                        outputs = self.model(inputs)

                        # Calculate loss
                        loss += self.criterion(outputs, masks).item()# + miou_loss(outputs, masks).item()
                        miou += miou_coef(outputs, masks).item()

                        # Visualization
                        self.show_image(outputs, masks)

            loss /= dataset_len
            miou /= dataset_len
            return loss, miou
        else: raise ValueError('The length of dataset must be at least 1.')

    def train(self):
        # Record writer
        make_folder(self.csv_path)

        with open(f'{self.csv_path}{csv_name}', 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, columns)
            writer.writeheader()

            dataset_len = len(self.train_set)
            if dataset_len > 0:
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
                            loss = self.criterion(outputs, masks)# + miou_loss(outputs, masks)
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

                    train_loss /= dataset_len
                    train_miou /= dataset_len
                    print(f'Epoch: {epoch}, Train_loss: {train_loss}, Train_loss_mini: {train_loss_mini}, Train_loss_micro: {train_loss_micro}, Train_miou: {train_miou}')

                    # Evaluation
                    val_loss, val_miou = self.eval(self.val_set)
                    self.scheduler.step(val_miou)
                    print(f'Epoch: {epoch}, Val_loss: {val_loss}, Val_miou: {val_miou}')

                    # Checkpoint save
                    if epoch % self.checkpoint_step == 0:
                        self.save(self.checkpoint_path, f'{self.checkpoint_path}epoch_{epoch}.pth')

                    # Record
                    writer.writerow({
                        columns[0]: epoch,
                        columns[1]: train_loss,
                        columns[2]: train_loss_mini,
                        columns[3]: train_loss_micro,
                        columns[4]: train_miou,
                        columns[5]: val_loss,
                        columns[6]: val_miou
                    })
                    self.tensorboard.add_scalar(columns[1], train_loss, epoch)
                    self.tensorboard.add_scalar(columns[2], train_loss_mini, epoch)
                    self.tensorboard.add_scalar(columns[3], train_loss_micro, epoch)
                    self.tensorboard.add_scalar(columns[4], train_miou, epoch)
                    self.tensorboard.add_scalar(columns[5], val_loss, epoch)
                    self.tensorboard.add_scalar(columns[6], val_miou, epoch)

                # Test
                test_loss, test_miou = self.eval(self.test_set)
                print(f'Test_loss: {test_loss}, Test_miou: {test_miou}')

                # Model save
                self.save(self.model_path, f'{self.model_path}model.pth')

                # Total train time
                end_time = time.time()
                total_time = end_time - start_time
                print(f'Total Train time: {round(total_time, 3)}s, {round(total_time/60, 3)}min, {round((total_time/60)/60, 3)}h.')

                self.tensorboard.close()
            else: raise ValueError('The length of dataset must be at least 1.')


class Plot():
    def __init__(
            self,
            csv_path: str,
    ) -> None:
        self.csv = pd.read_csv(csv_path + csv_name)

    def show_plot(self) -> None:
        x = self.csv[columns[0]]

        plt.figure()
        for i in range(1, len(columns)):
            plt.plot(x, self.csv[columns[i]])

        plt.xlabel(columns[0])
        plt.legend()
        plt.show()

