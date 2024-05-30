from unet import UNet
from torch import device
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from utils import makedirs
import torch
import matplotlib.pyplot as plt


csv_name = 'train_logg.csv'
csv_path = 'csv/'
column = [
    'Epoch',
    'Train_loss',
    'Train_loss_mini',
    'Train_loss_micro',
    'Train_miou',
    'Val_loss',
    'Val_miou'
]
checkpoint_path = 'model/checkpoint/'
model_path = 'model/'


class Train():
    def __init__(
            self,
            model: UNet,
            dataset: dict,
            lr: float,
            device: device,
            epochs: int,
            accumulation_step: int,
            checkpoint_step: int,
            show: float = 0.0
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.epochs = epochs
        self.accumulation_step = accumulation_step
        self.checkpoint_step = checkpoint_step
        self.show = show

        self.train_set = dataset['train']
        self.val_set = dataset['val']
        self.test_set = dataset['test']

        self.optim = optim.Adam(model.parameters(), lr)
        self.criterion = nn.BCEWithLogitsLoss().to(device)
        self.scaler = GradScaler()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'max', patience=5)

        self.tesnorboard = SummaryWriter()

    def save_model(self, save_path: str, save_name: str) -> None:
        makedirs(save_path)
        torch.save(self.model.state_dict(), save_name)
        print(f'Model saved at {save_name}.')