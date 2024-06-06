from torch import device, cuda
from utils import load_config
from unet import EnsembleUNet, UNet
from dataset import Augmentation, SegmentationDataLoader, SegmentationDataset
from train import Trainer


CONFIG_PATH = 'config/config.yaml'
DEVICE = device('cuda' if cuda.is_available() else 'cpu')


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
        if model_cfg['ensemble'] > 1:
            unet_models = []
            for i in range(model_cfg['ensemble']):
                unet_models.append(model)
            model = EnsembleUNet(unet_models)


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
             p=augmentation_cfg['p'],
             normalize=augmentation_cfg['normalize']
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


        # Trainer
        trainer_cfg = cfg['trainer']
        trainer = Trainer(
              model=model,
              dataset=dataset.get_loader(debug=True),
              lr=trainer_cfg['lr'],
              device=DEVICE,
              epochs=trainer_cfg['epochs'],
              accumulation_step=trainer_cfg['accumulation_step'],
              checkpoint_step=trainer_cfg['checkpoint_step'],
              show_time=trainer_cfg['show_time']
        )
        trainer.train(trainer_cfg['checkpoint_path'], trainer_cfg['model_path'])

