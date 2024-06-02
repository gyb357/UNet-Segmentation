from torch import device, cuda
from utils import load_config
from unet import UNet
from dataset import Augmentation, SegmentationDataLoader, SegmentationDataset
from train import Trainer
from test import Tester


config_path = 'config/config.yaml'


dev = device('cuda' if cuda.is_available() else 'cpu')


def main() -> None:
    config = load_config(config_path)

    train_config = config['train']
    test_config = config['test']
    plot_config = config['plot']

    if train_config['train'] or test_config['test']:
        model_config = config['model']
        model = UNet(
            channels=model_config['channels'],
            num_classes=model_config['num_classes'],
            backbone=model_config['backbone'],
            pretrained=model_config['pretrained'],
            freeze_grad=model_config['freeze_grad'],
            kernel_size=model_config['kernel_size'],
            bias=model_config['bias'],
            norm=model_config['norm'],
            dropout=model_config['dropout']
        )
        # print(model)

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

        if train_config['train']:
            dataloader_config = config['dataloader']
            dataloader = SegmentationDataLoader(
                image_path=dataloader_config['image_path'],
                mask_path=dataloader_config['mask_path'],
                extension=dataloader_config['extension'],
                num_images=dataloader_config['num_images'],
                augmentation=augmentation
            )

            dataset_config = config['dataset']
            dataset = SegmentationDataset(
                dataset_loader=dataloader,
                dataset_dict=dataset_config['dataset_dict'],
                batch_size=dataset_config['batch_size'],
                shuffle=dataset_config['shuffle'],
                num_workers=dataset_config['num_workers'],
                pin_memory=dataset_config['pin_memory']
            )

            train = Trainer(
                model=model,
                dataset=dataset.get_loader(debug=True),
                lr=train_config['lr'],
                device=dev,
                epochs=train_config['epochs'],
                accumulation_step=train_config['accumulation_step'],
                checkpoint_step=train_config['checkpoint_step'],
                show_time=train_config['show_time'],
                show_plt=train_config['show_plt']
            )
            train.train()

            if test_config['test']:
                test = Tester(
                    model=model,
                    device=dev,
                    augmentation=augmentation,
                    threshold=test_config['thres_hold'],
                    model_path=test_config['model_path'],
                    image_path=test_config['image_path']
                )
                test.test()


if __name__ == '__main__':
    main()

