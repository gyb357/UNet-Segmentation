import json
from torch import device, cuda
from dataset import Augmentation, SegmentationDataset
from torch.utils.data import random_split, DataLoader
from model import UNet
import torch.nn as nn
from train import Train
import torch.optim as optim
import time


def load_config(path: str):
    with open(path, 'r') as f:
        return json.load(f)


config_path = 'config/config.json'
config = load_config(config_path)


# Device
DEVICE = device(config['DEVICE'] if cuda.is_available() else 'cpu')


if __name__ == '__main__':
    augmentation = Augmentation(
        channels=config['CHANNELS'],
        resize=config['RESIZE'],
        crop=config['CROP'],
        hflip=config['FLIP'],
        vflip=config['FLIP']
    )
    ubiris_dataset = SegmentationDataset(
        image_path=config['UBIRIS_IMAGE_PATH'],
        mask_path=config['UBIRIS_MASK_PATH'],
        extension=config['UBIRIS_EXTENSION'],
        num_images=config['NUM_IMAGES'],
        augmentation=augmentation
    )
    casia_dataset = SegmentationDataset(
        image_path=config['CASIA_IMAGE_PATH'],
        mask_path=config['CASIA_MASK_PATH'],
        extension=config['CASIA_EXTENSION'],
        num_images=config['NUM_IMAGES'],
        augmentation=augmentation
    )


    dataset_len = len(ubiris_dataset) + len(casia_dataset)
    train_len = int(config['DATASET_LENTH']['train']*dataset_len)
    valid_len = int(config['DATASET_LENTH']['valid']*dataset_len)
    test_len = dataset_len - train_len - valid_len
    total_dataset = ubiris_dataset + casia_dataset
    print(f'ubiris_dataset: {len(ubiris_dataset)}, casia_dataset: {len(casia_dataset)}, total_dataset: {len(total_dataset)}')


    train_dataset, valid_dataset, test_dataset = random_split(total_dataset, [train_len, valid_len, test_len])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=config['SHUFFLE'],
        num_workers=config['NUM_WORKERS'],
        pin_memory=config['PIN_MEMORY']
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=config['SHUFFLE'],
        num_workers=config['NUM_WORKERS'],
        pin_memory=config['PIN_MEMORY']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=config['SHUFFLE'],
        num_workers=config['NUM_WORKERS'],
        pin_memory=config['PIN_MEMORY']
    )
    print(f'train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}, test_loader: {len(test_loader)}')


    model = UNet(
        in_channels=config['CHANNELS'],
        out_channels=config['CHANNELS'],
        filter=config['FILTER'],
        kernel_size=config['KERNEL_SIZE'],
        bias=config['BIAS'],
        dropout=config['DROPOUT'],
        activation=nn.Sigmoid(),
        checkpoint=config['CHECKPOINT']
    ).to(DEVICE)
    model.init_weights(model)
    # print(model)


    train = Train(
        model=model,
        criterion=nn.BCEWithLogitsLoss().to(DEVICE),
        optim=optim.Adam(model.parameters(), lr=config['LR']),
        train_set=train_loader,
        valid_set=valid_loader,
        test_set=test_loader,
        epochs=config['EPOCHS'],
        accumulation=config['ACCUMULATION'],
        device=DEVICE,
        show=config['SHOW'],
        csv_path=config['CSV_PATH'],
        checkpoint_path=config['CHECKPOINT_PATH'],
        model_path=config['MODEL_PATH']
    )
    start = time.time()
    train.fit()
    end = time.time()


    test_loss, test_dice_loss = train.test(test_loader)
    print(f'test_loss: {test_loss}, test_dice_loss: {test_dice_loss}')
    print(f'Training is complete: {end - start} s')

