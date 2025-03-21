import torch.nn as nn
import torch
from utils import load_config
from model.unet import UNet
from dataset.dataset import MaskDatasetGenerator, Augmentation, SegmentationDataset, SegmentationDataLoader
from train.train import Trainer


if __name__ == '__main__':
    config = load_config("config/config.yaml")
    config_task = config['tasks']
    config_train = config_task['train']
    config_model = config_task['test']

    if config_task['train'] or config_task['test']:
        config_model = config['model']
        model = UNet(
            channels=config_model['channels'],
            num_classes=config_model['num_classes'],
            backbone=config_model['backbone'],
            pretrained=config_model['pretrained'],
            freeze_backbone=config_model['freeze_backbone'],
            bias=config_model['bias'],
            normalize=config_model['normalize'],
            dropout=config_model['dropout'],
            init_weights=config_model['init_weights']
        )

    if config_task['train']:
        config_mask_dataset = config['mask_dataset']
        mask_dataset_label_path = config_mask_dataset['label_path']
        mask_dataset_mask_path = config_mask_dataset['mask_path']

        if mask_dataset_label_path != '' or mask_dataset_mask_path != '':
            mask_dataset_generator = MaskDatasetGenerator(
                label_path=mask_dataset_label_path,
                mask_path=mask_dataset_mask_path,
                mask_size=config_mask_dataset['mask_size'],
                mask_extension=config_mask_dataset['mask_extension'],
                mask_fill=config_mask_dataset['mask_fill']
            )
            mask_dataset_generator()

        config_augmentation = config['augmentation']
        augmentation = Augmentation(
            channels=config_model['channels'],
            resize=tuple(config_augmentation['resize']),
            hflip=config_augmentation['hflip'],
            vflip=config_augmentation['vflip'],
            rotate=config_augmentation['rotate'],
            saturation=config_augmentation['saturation'],
            brightness=config_augmentation['brightness'],
            factor=config_augmentation['factor'],
            p=config_augmentation['p']
        )

        config_segmentation_dataset = config['segmentation_dataset']
        segmentation_dataset = SegmentationDataset(
            image_path=config_segmentation_dataset['image_path'],
            mask_path=config_segmentation_dataset['mask_path'],
            extension=config_segmentation_dataset['extension'],
            num_images=config_segmentation_dataset['num_images'],
            augmentation=augmentation
        )

        config_segmentation_dataloader = config['segmentation_dataloader']
        segmentation_dataloader = SegmentationDataLoader(
            dataset=segmentation_dataset,
            dataset_split=config_segmentation_dataloader['dataset_split'],
            batch_size=config_segmentation_dataloader['batch_size'],
            shuffle=config_segmentation_dataloader['shuffle'],
            num_workers=config_segmentation_dataloader['num_workers'],
            pin_memory=config_segmentation_dataloader['pin_memory']
        )

        config_trainer = config['trainer']
        trainer_loss = config_trainer['loss']
        if trainer_loss == 'BCEWithLogitsLoss':
            loss_func = nn.BCEWithLogitsLoss()
        elif trainer_loss == 'CrossEntropyLoss':
            loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Invalid loss function: {trainer_loss}. Please choose from ['BCEWithLogitsLoss', 'CrossEntropyLoss']")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = Trainer(
            model=model,
            dataloader=segmentation_dataloader.get_loader(),
            lr=config_trainer['lr'],
            loss=loss_func,
            loss_coefficient=config_trainer['loss_coefficient'],
            device=device,
            epochs=config_trainer['epochs'],
            accumulation_step=config_trainer['accumulation_step'],
            checkpoint_step=config_trainer['checkpoint_step'],
            show_time=config_trainer['show_time'],
            num_classes=config_model['num_classes'],
            early_stopping_patience=config_trainer['early_stopping_patience'],
            mixed_precision=config_trainer['mixed_precision'],
            weight_decay=config_trainer['weight_decay']
        )
        trainer.fit(
            csv_path='csv/',
            csv_name='train_log.csv',
            checkpoint_path='model/checkpoint',
            model_path='model'
        )

    if config_task['test']:
        pass

