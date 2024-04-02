from utils import load_config
from torch import device, cuda
from dataset import Augmentation, SegmentationDataset
from torch.utils.data import random_split, DataLoader
from model import UNet
from train import Train
import torch.optim as optim


config = load_config('config/config.json')


dev = device(config['device'] if cuda.is_available() else 'cpu')


if __name__ == '__main__':
    augmentation = Augmentation(
        channels = config['channels'],
        crop     = config['crop'],
        resize   = config['resize'],
        hflip    = config['flip'],
        vflip    = config['flip']
    )
    ubiris_dataset = SegmentationDataset(
        image_path   = config['ubiris_image_path'],
        mask_path    = config['ubiris_mask_path'],
        extension    = config['ubiris_extension'],
        num_images   = config['num_images'],
        augmentation = augmentation
    )
    casia_dataset = SegmentationDataset(
        image_path   = config['casia_image_path'],
        mask_path    = config['casia_mask_path'],
        extension    = config['casia_extension'],
        num_images   = config['num_images'],
        augmentation = augmentation
    )


    dataset_len = len(ubiris_dataset) + len(casia_dataset)
    train_len = int(config['dataset_lenth']['train']*dataset_len)
    valid_len = int(config['dataset_lenth']['valid']*dataset_len)
    test_len = dataset_len - train_len - valid_len
    total_dataset = ubiris_dataset + casia_dataset
    print(f'ubiris_dataset: {len(ubiris_dataset)}, casia_dataset: {len(casia_dataset)}, total_dataset: {len(total_dataset)}')


    train_dataset, valid_dataset, test_dataset = random_split(total_dataset, [train_len, valid_len, test_len])
    train_loader = DataLoader(
        train_dataset,
        batch_size  = config['batch_size'],
        shuffle     = config['shuffle'],
        num_workers = config['num_workers'],
        pin_memory  = config['pin_memory']
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size  = config['batch_size'],
        shuffle     = config['shuffle'],
        num_workers = config['num_workers'],
        pin_memory  = config['pin_memory']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = config['batch_size'],
        shuffle     = config['shuffle'],
        num_workers = config['num_workers'],
        pin_memory  = config['pin_memory']
    )
    print(f'train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}, test_loader: {len(test_loader)}')


    model = UNet(
        in_channels         = config['channels'],
        out_channels        = config['channels'],
        filter              = config['filter'],
        kernel_size         = config['kernel_size'],
        bias                = config['bias'],
        dropout             = config['dropout'],
        initialize_weights  = config['initialize_weights'],
        gradient_checkpoint = config['gradient_checkpoint']
    ).to(dev)
    # print(model)


    train = Train(
        model             = model,
        optimizer         = optim.Adam(model.parameters(), lr=config['lr']),
        train_set         = train_loader,
        valid_set         = valid_loader,
        test_set          = test_loader,
        epochs            = config['epochs'],
        accumulation_step = config['accumulation_step'],
        checkpoint_step   = config['checkpoint_step'],
        device            = dev,
        show              = config['show'],
        csv_path          = config['csv_path'],
        checkpoint_path   = config['checkpoint_path'],
        model_path        = config['model_path']
    ).train()

