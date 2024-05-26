from unet import UNet
from torch import device
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter


class Train():
    csv_path: str = 'csv'
    checkpoint_path: str = 'model/checkpoint'
    model_path: str = 'model'

    def __init__(
            self,
            model: UNet,
            dataset: dict,
            learning_rate: float,
            epochs: int,
            accumulation_step: int,
            checkpoint_step: int,
            device: device,
            show: float
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.accumulation_step = accumulation_step
        self.checkpoint_step = checkpoint_step
        self.device = device
        self.show = show

        self.train_set = dataset['train']
        self.val_set = dataset['val']
        self.test_set = dataset['test']

        self.optim = optim.Adam(model.parameters(), learning_rate)
        self.criterion = nn.BCEWithLogitsLoss().to(device)
        self.scaler = GradScaler()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'max', patience=5)

        self.tensorboard = SummaryWriter()