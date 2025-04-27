if __name__ == "__main__":
    # Imports
    import os
    import torch.nn as nn
    from utils import load_config, get_model_list
    from model.unet import UNet, UNet2Plus, UNet3Plus, EnsembleUNet
    from dataset.dataset import MaskDatasetGenerator, Augmentation, SegmentationDataset, SegmentationDataLoader
    from train.train import Trainer


    # Configs
    cfg = load_config("config/config.yaml")

    # Task
    cfg_task = cfg["task"]
    task_train = cfg_task["train"]
    task_test = cfg_task["test"]

    # Model
    if task_train or task_test:
        cfg_model = cfg["model"]
        model_ensemble = cfg_model["ensemble"]

        # Get model list
        model_list = get_model_list(model_ensemble)

        # Ensemble model
        if model_list:
            model = EnsembleUNet(
                model_names=model_ensemble,
                channels=cfg_model["channels"],
                num_classes=cfg_model["num_classes"],
                backbone=cfg_model["backbone"],
                pretrained=cfg_model["pretrained"],
                freeze_backbone=cfg_model["freeze_backbone"],
                bias=cfg_model["bias"],
                normalize=cfg_model["normalize"],
                dropout=cfg_model["dropout"],
                init_weights=cfg_model["init_weights"],
                deep_supervision=cfg_model["deep_supervision"],
                cgm=cfg_model["cgm"]
            )

        # Single model
        else:
            model_name = cfg_model["name"]
            if model_name == 'unet':
                model = UNet
            elif model_name == 'unet2plus':
                model = UNet2Plus
            elif model_name == 'unet3plus':
                model = UNet3Plus
            else:
                raise ValueError(f"Model {model_name} not recognized. Please check the model name in the config file.")
            
            model = model(
                channels=cfg_model["channels"],
                num_classes=cfg_model["num_classes"],
                backbone=cfg_model["backbone"],
                pretrained=cfg_model["pretrained"],
                freeze_backbone=cfg_model["freeze_backbone"],
                bias=cfg_model["bias"],
                normalize=cfg_model["normalize"],
                dropout=cfg_model["dropout"],
                init_weights=cfg_model["init_weights"],
                deep_supervision=cfg_model["deep_supervision"],
                cgm=cfg_model["cgm"]
            )

        # Print model parameters
        parameters = model._get_parameters()
        print(f"Total Model Parameters: {parameters}, ({parameters / 1e6:.2f} M)")

    # MaskDatasetGenerator
    if task_train:
        cfg_mask_dataset = cfg["mask_dataset"]
        if not os.path.exists(cfg_mask_dataset["mask_dir"]):
            print(f"Mask directory {cfg_mask_dataset['mask_dir']} does not exist. Creating it.")

            mask_dataset_generator = MaskDatasetGenerator(
                cfg_mask_dataset["label_dir"],
                cfg_mask_dataset["mask_dir"],
                cfg_mask_dataset["mask_size"],
                cfg_mask_dataset["mask_extension"],
                cfg_mask_dataset["mask_fill"]
            ).generate()
            
        # Augmentation
        cfg_augmentation = cfg["augmentation"]
        augmentation = Augmentation(
            cfg_augmentation["channels"],
            cfg_augmentation["resize"],
            cfg_augmentation["hflip"],
            cfg_augmentation["vflip"],
            cfg_augmentation["rotate"],
            cfg_augmentation["saturation"],
            cfg_augmentation["brightness"],
            cfg_augmentation["factor"],
            cfg_augmentation["p"],
        )

        # SegmentationDataset
        cfg_dataset = cfg["segmentation_dataset"]
        dataset = SegmentationDataset(
            cfg_dataset["image_dir"],
            cfg_dataset["mask_dir"],
            cfg_dataset["extension"],
            cfg_dataset["num_images"],
            augmentation=augmentation
        )

        # SegmentationDataLoader
        cfg_dataloader = cfg["segmentation_dataLoader"]
        dataloader = SegmentationDataLoader(
            dataset=dataset,
            split=cfg_dataloader["split"],
            batch_size=cfg_dataloader["batch_size"],
            shuffle=cfg_dataloader["shuffle"],
            num_workers=cfg_dataloader["num_workers"],
            pin_memory=cfg_dataloader["pin_memory"],
            seed=cfg_dataloader["seed"]
        )

        # Trainer
        cfg_trainer = cfg["trainer"]
        # Loss
        cfg_loss_fn = cfg_trainer["loss_fn"]
        if cfg_loss_fn == 'BCEWithLogitsLoss':
            loss_fn = nn.BCEWithLogitsLoss()
        elif cfg_loss_fn == 'CrossEntropyLoss':
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Loss function {cfg_loss_fn} not recognized. Please check the loss function in the config file.")

        trainer = Trainer(
            device=cfg_trainer["device"],
            model=model,
            loss_fn=loss_fn,
            metrics_fn=cfg_trainer["metrics_fn"],
            num_classes=cfg_model["num_classes"],
            dataloader=dataloader.get_loaders(),
            lr=cfg_trainer["lr"],
            weight_decay=cfg_trainer["weight_decay"],
            mixed_precision=cfg_trainer["mixed_precision"],
            epochs=cfg_trainer["epochs"],
            accumulation_step=cfg_trainer["accumulation_step"],
            checkpoint_step=cfg_trainer["checkpoint_step"],
            early_stopping_patience=cfg_trainer["early_stopping_patience"]
        ).fit()

