from torch import device, cuda
import torch.nn as nn
from utils import load_config
from unet import UNet, EnsembleUNet
from torchsummary import summary
from dataset import Augmentation, SegmentationDataLoader, SegmentationDataset
from train import Trainer
from test import Tester


CONFIG_PATH = 'config/config.yaml'
DEVICE = device('cuda' if cuda.is_available() else 'cpu')
LOSS = nn.BCEWithLogitsLoss().to(DEVICE) # LOSS = nn.CrossEntropyLoss().to(DEVICE)


if __name__ == '__main__':
    config = load_config(CONFIG_PATH)
    trainer_config = config['trainer']
    tester_config = config['tester']

    # Define model
    if trainer_config['train'] or tester_config['test']:
        model_config = config['model']
        model = UNet(
            channels=model_config['channels'],
            num_classes=model_config['num_classes'],
            backbone_name=model_config['backbone_name'],
            pretrained=model_config['pretrained'],
            freeze_grad=model_config['freeze_grad'],
            kernel_size=model_config['kernel_size'],
            bias=model_config['bias'],
            normalize=model_config['normalize'],
            dropout=model_config['dropout'],
            init_weights=model_config['init_weights'],
        )
        if model_config['ensemble']:
            unet_models = []

            for ensemble_config in model_config['ensemble']:
                unet = UNet(
                    channels=ensemble_config['channels'],
                    num_classes=ensemble_config['num_classes'],
                    backbone_name=ensemble_config['backbone_name'],
                    pretrained=ensemble_config['pretrained'],
                    freeze_grad=ensemble_config['freeze_grad'],
                    kernel_size=ensemble_config['kernel_size'],
                    bias=ensemble_config['bias'],
                    normalize=ensemble_config['normalize'],
                    dropout=ensemble_config['dropout'],
                    init_weights=ensemble_config['init_weights'],
                )
                unet_models.append(unet)
            model = EnsembleUNet(unet_models)
        model.to(DEVICE)

        # Define Augmentation
        augmentation_config = config['augmentation']
        augmentation = Augmentation(
             channels=augmentation_config['channels'],
             resize=augmentation_config['resize'],
             hflip=augmentation_config['hflip'],
             vflip=augmentation_config['vflip'],
             rotate=augmentation_config['rotate'],
             saturation=augmentation_config['saturation'],
             brightness=augmentation_config['brightness'],
             factor=augmentation_config['factor'],
             p=augmentation_config['p']
        )

        # Define dataloader
        dataloader_cfg = config['dataloader']
        dataloader = SegmentationDataLoader(
            image_path=dataloader_cfg['image_path'],
            mask_path=dataloader_cfg['mask_path'],
            extension=dataloader_cfg['extension'],
            num_images=dataloader_cfg['num_images'],
            augmentation=augmentation
        )

        # Define dataset
        dataset_cfg = config['dataset']
        dataset = SegmentationDataset(
            dataset_loader=dataloader,
            dataset_split=dataset_cfg['dataset_split'],
            batch_size=dataset_cfg['batch_size'],
            shuffle=dataset_cfg['shuffle'],
            num_workers=dataset_cfg['num_workers'],
            pin_memory=dataset_cfg['pin_memory']
        )
        summary(
            model,
            input_size=(augmentation_config['channels'], augmentation_config['resize'][0], augmentation_config['resize'][1]),
            batch_size=dataset_cfg['batch_size']
        )

        # Define trainer
        if trainer_config['train']:
            trainer = Trainer(
                  model=model,
                  dataset=dataset.get_loader(debug=True),
                  lr=trainer_config['lr'],
                  loss=LOSS,
                  device=DEVICE,
                  epochs=trainer_config['epochs'],
                  accumulation_step=trainer_config['accumulation_step'],
                  checkpoint_step=trainer_config['checkpoint_step'],
                  show_time=trainer_config['show_time']
            )
            trainer.train(
                csv_path=trainer_config['csv_path'],
                csv_name=trainer_config['csv_name'],
                checkpoint_path=trainer_config['checkpoint_path'],
                model_path=trainer_config['model_path']
            )

        # Define tester
        if tester_config['test']:
            tester = Tester(
                model=model,
                device=DEVICE,
                augmentation=augmentation,
                threshold=tester_config['threshold']
            )
            tester.test(
                model_path=tester_config['model_path'],
                image_path=tester_config['image_path']
            )

