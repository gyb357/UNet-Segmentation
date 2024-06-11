from torch import device, cuda
import torch.nn as nn
from utils import load_config
from unet import UNet, EnsembleUNet
# from torchsummary import summary
from dataset import Augmentation, SegmentationDataLoader, SegmentationDataset
from train import Trainer
from test import Tester


CONFIG_PATH = 'config/config.yaml'
DEVICE = device('cuda' if cuda.is_available() else 'cpu')
LOSS = nn.BCEWithLogitsLoss().to(DEVICE)


if __name__ == '__main__':
    cfg = load_config(CONFIG_PATH)
    trainer_cfg = cfg['trainer']
    tester_cfg = cfg['tester']
    
    if trainer_cfg['train'] or tester_cfg['test']:
        # Model
        model_cfg = cfg['model']
        model = UNet(
            channels=model_cfg['channels'],
            num_classes=model_cfg['num_classes'],
            backbone=model_cfg['backbone'],
            pretrained=model_cfg['pretrained'],
            freeze_grad=model_cfg['freeze_grad'],
            kernel_size=model_cfg['kernel_size'],
            bias=model_cfg['bias'],
            norm=model_cfg['norm'],
            dropout=model_cfg['dropout'],
            init_weights=model_cfg['init_weights']
        )
        if model_cfg['ensemble'] is not None:
            unet_models = []

            for ensemble_cfg in model_cfg['ensemble']:
               print(ensemble_cfg)
               unet = UNet(
                    channels=ensemble_cfg['channels'],
                    num_classes=ensemble_cfg['num_classes'],
                    backbone=ensemble_cfg['backbone'],
                    pretrained=ensemble_cfg['pretrained'],
                    freeze_grad=ensemble_cfg['freeze_grad'],
                    kernel_size=ensemble_cfg['kernel_size'],
                    bias=ensemble_cfg['bias'],
                    norm=ensemble_cfg['norm'],
                    dropout=ensemble_cfg['dropout'],
                    init_weights=ensemble_cfg['init_weights']
               )
               unet_models.append(unet)
            model = EnsembleUNet(unet_models)
        model.to(DEVICE)

        # Augmentation
        augmentation_cfg = cfg['augmentation']
        augmentation = Augmentation(
             channels=augmentation_cfg['channels'],
             resize=augmentation_cfg['resize'],
             hflip=augmentation_cfg['hflip'],
             vflip=augmentation_cfg['vflip'],
             rotate=augmentation_cfg['rotate'],
             saturation=augmentation_cfg['saturation'],
             brightness=augmentation_cfg['brightness'],
             factor=augmentation_cfg['factor'],
             p=augmentation_cfg['p']
        )

        # Dataloader
        dataloader_cfg = cfg['dataloader']
        dataloader = SegmentationDataLoader(
            image_path=dataloader_cfg['image_path'],
            mask_path=dataloader_cfg['mask_path'],
            extension=dataloader_cfg['extension'],
            num_images=dataloader_cfg['num_images'],
            augmentation=augmentation
        )

        # Dataset
        dataset_cfg = cfg['dataset']
        dataset = SegmentationDataset(
            dataset_loader=dataloader,
            dataset_split=dataset_cfg['dataset_split'],
            batch_size=dataset_cfg['batch_size'],
            shuffle=dataset_cfg['shuffle'],
            num_workers=dataset_cfg['num_workers'],
            pin_memory=dataset_cfg['pin_memory']
        )

        # summary(
        #     model,
        #     input_size=(augmentation_cfg['channels'], augmentation_cfg['resize'][0], augmentation_cfg['resize'][1]),
        #     batch_size=dataset_cfg['batch_size']
        # )

        # Trainer
        if trainer_cfg['train']:
            trainer = Trainer(
                  model=model,
                  dataset=dataset.get_loader(debug=True),
                  lr=trainer_cfg['lr'],
                  loss=LOSS,
                  device=DEVICE,
                  epochs=trainer_cfg['epochs'],
                  accumulation_step=trainer_cfg['accumulation_step'],
                  checkpoint_step=trainer_cfg['checkpoint_step'],
                  show_time=trainer_cfg['show_time']
            )
            trainer.train(
                csv_path=trainer_cfg['csv_path'],
                csv_name=trainer_cfg['csv_name'],
                checkpoint_path=trainer_cfg['checkpoint_path'],
                model_path=trainer_cfg['model_path']
            )

        # Tester
        if tester_cfg['test']:
            tester = Tester(
                model=model,
                device=DEVICE,
                augmentation=augmentation,
                threshold=tester_cfg['threshold']
            )
            tester.test(
                model_path=tester_cfg['model_path'],
                image_path=tester_cfg['image_path']
            )

