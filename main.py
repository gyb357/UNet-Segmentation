from torch import device, cuda
from dataset import Augmentation, SegmentationDataset
from torch.utils.data import random_split, DataLoader
from model import UNet
from train import Train
# import torch.nn as nn
from loss import DiceBCELoss, IoULoss
import torch.optim as optim


# Augmentation
CHANNELS = 1
RESIZE = (256, 256)
CROP = (192, 192)
FLIP = True
DEVICE = device('cuda' if cuda.is_available() else 'cpu')
GPU = False
# SegmentationDataset
UBIRIS_IMAGE_PATH = 'dataset/ubiris/image'
UBIRIS_MASK_PATH = 'dataset/ubiris/mask'
UBIRIS_EXTENSION = '.tiff'
CASIA_IMAGE_PATH = 'dataset/casia/image'
CASIA_MASK_PATH = 'dataset/casia/mask'
CASIA_EXTENSION = '.png'
NUM_IMAGES = 2250
# DataLoader
DATASET_LEN_COEF = {
    'train': 0.7,
    'valid': 0.15,
    'test': 0.1
}
BATCH_SIZE = 32
SHUFFLE = True
NUM_WORKERS = 8
PIN_MEMORY = True
# UNet
CHANNELS = 1
FILTER = 64
DROPOUT = 0.2
# Train
LEARNING_RATE = 1e-4
EPOCHS = 100
ACCUMULATION_STEPS = 2
SHOW_TIME = 1
CSV_FILENAME = 'csv/iris_train.csv'
MODEL_FILENAME = 'model/model2.pth'


if __name__ == '__main__':
    augmentation = Augmentation(
        channels=CHANNELS,
        resize=RESIZE,
        crop=CROP,
        hflip=FLIP,
        vflip=FLIP,
        device=DEVICE,
        gpu=GPU
    )
    ubiris_dataset = SegmentationDataset(
        origin_path=UBIRIS_IMAGE_PATH,
        mask_path=UBIRIS_MASK_PATH,
        extension=UBIRIS_EXTENSION,
        num_images=NUM_IMAGES,
        augmentation=augmentation
    )
    casia_dataset = SegmentationDataset(
        origin_path=CASIA_IMAGE_PATH,
        mask_path=CASIA_MASK_PATH,
        extension=CASIA_EXTENSION,
        num_images=NUM_IMAGES,
        augmentation=augmentation
    )
    print(f'ubiris_dataset: {len(ubiris_dataset)}, casia_dataset: {len(casia_dataset)}')


    dataset_len = len(ubiris_dataset) + len(casia_dataset)
    train_len = int(DATASET_LEN_COEF['train']*dataset_len)
    valid_len = int(DATASET_LEN_COEF['valid']*dataset_len)
    test_len = dataset_len - train_len - valid_len
    total_dataset = ubiris_dataset + casia_dataset


    train_dataset, valid_dataset, test_dataset = random_split(total_dataset, [train_len, valid_len, test_len])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    print(f'train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}, test_loader: {len(test_loader)}')


    model = UNet(
        in_channels=CHANNELS,
        filter=FILTER,
        out_channels=CHANNELS,
        kernel_size=3,
        bias=True,
        dropout=DROPOUT
    ).to(DEVICE)
    # print(model)


    train = Train(
        model=model,
        criterion=IoULoss().to(DEVICE),
        optim=optim.Adam(model.parameters(), lr=LEARNING_RATE),
        train_set=train_loader,
        valid_set=valid_loader,
        test_set=test_loader,
        epochs=EPOCHS,
        accumulation=ACCUMULATION_STEPS,
        device=DEVICE,
        show=SHOW_TIME,
        csv_path=CSV_FILENAME,
        model_path=MODEL_FILENAME
    )
    train.fit()

