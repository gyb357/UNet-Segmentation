from utils import load_config
from torch import device, cuda
from dataset import Augmentation, SegmentationDataset, Dataset
from model import UNet
from train import Train, Plot
from test import Test


config = load_config('config/config.json')


dev = device('cuda' if cuda.is_available() else 'cpu')


if __name__ == '__main__':
    config_train = config['train']
    config_test = config['test']
    config_plot = config['plot']

    if config_train['train'] == True or config_test['test'] == True:
        config_model = config['model']
        model = UNet(
            in_channels  = config_model['in_channels'],
            out_channels = config_model['out_channels'],
            filter       = config_model['filter'],
            kernel_size  = config_model['kernel_size'],
            bias         = config_model['bias'],
            dropout      = config_model['dropout'],
            batch_normal = config_model['batch_normal'],
            init_weights = config_model['init_weights']
        ).to(dev)
        # print(model)

        config_aug = config['augmentation']
        augmentation = Augmentation(
            channels = config_aug['channels'],
            crop     = config_aug['crop'],
            resize   = config_aug['resize'],
            hflip    = config_aug['flip'],
            vflip    = config_aug['flip']
        )

    if config_train['train'] == True:
        config_seg = config['segmentation_dataset']
        ubiris_dataset = SegmentationDataset(
            image_path   = config_seg['ubiris_image_path'],
            mask_path    = config_seg['ubiris_mask_path'],
            extension    = config_seg['ubiris_extension'],
            num_images   = config_seg['ubiris_num_images'],
            augmentation = augmentation
        )
        ubipr_dataset = SegmentationDataset(
            image_path   = config_seg['ubipr_image_path'],
            mask_path    = config_seg['ubipr_mask_path'],
            extension    = config_seg['ubipr_extension'],
            num_images   = config_seg['ubipr_num_images'],
            augmentation = augmentation
        )
    
        config_data = config['dataset']
        dataset = Dataset(
            dataset       = ubiris_dataset + ubipr_dataset,
            dataset_split = config_data['dataset_split'],
            batch_size    = config_data['batch_size'],
            shuffle       = config_data['shuffle'],
            num_workers   = config_data['num_workers'],
            pin_memory    = config_data['pin_memory']
        )
        
        train = Train(
            model             = model,
            dataset           = dataset.get_dataloader(),
            lr                = config_train['lr'],
            epochs            = config_train['epochs'],
            accumulation_step = config_train['accumulation_step'],
            checkpoint_step   = config_train['checkpoint_step'],
            device            = dev,
            show              = config_train['show'],
            csv_path          = config_train['csv_path'],
            checkpoint_path   = config_train['checkpoint_path'],
            model_path        = config_train['model_path']
        )
        train.train()

    if config_plot['show'] == True:
        Plot(
            csv_path = config_plot['csv_path']
        ).show_plot()
        # tensorboard --logdir=

    if config_test['test'] == True:
        test = Test(
            model        = model,
            device       = dev,
            augmentation = augmentation,
            threshold    = config_test['threshold'],
            model_path   = config_test['model_path'],
            image_path   = config_test['image_path']
        )
        test.test()

