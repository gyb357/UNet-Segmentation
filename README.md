# Table of Contents

1. [Introduction](#Introduction)
     * UNet Implementations with ResNet Backbone
     * Purpose of the Project
     
2. [Architecture Overview](#Architecture-Overview)
     * UNet
     * UNet++
     * UNet3+

3. [Model Comparison](#Model-Comparison)

4. [Getting Started](#Getting-Started)
     * Check the Running Environment
     * Installation and Dependencies
     * Downloading Pre-Trained ResNet Weights
     * 
5. [Project Structure](#Project-Structure)


*****


# 📑Introduction

## UNet Implementations with ResNet Backbone
This repository implements multiple UNet-based architectures with ResNet backbones using PyTorch. The models leverage pretrained weights from ResNet to enhance feature extraction in the encoder parts of these architectures. The implementation includes UNet, UNet++, and UNet3+, all of which support ResNet backbones for improved performance on biomedical image segmentation tasks.

## Purpose of the Project
This project focuses on implementing and comparing various UNet architectures, integrating them with powerful ResNet backbones to create high-performance segmentation models. The implementation supports seamless switching between different UNet variants to find the optimal architecture for specific segmentation tasks.


*****


# 🔍Architecture Overview

![unet](assets/unet.png)

## UNet
The classic UNet architecture features a symmetric encoder-decoder structure with skip connections, allowing it to capture contextual information while preserving spatial details. This design makes UNet effective for general biomedical image segmentation tasks. The encoder progressively reduces spatial dimensions while extracting high-level features, and the decoder restores the original resolution using upsampling layers. Skip connections directly transfer fine-grained spatial information from the encoder to the decoder, helping to recover precise segmentation boundaries.

## UNet++
UNet++ redesigns the skip connections with a nested dense structure, effectively bridging the semantic gap between encoder and decoder features. This design introduces convolutional layers within the skip pathways, progressively refining features before passing them to the decoder. Additionally, UNet++ enabling more effective gradient propagation and improving segmentation accuracy.

## UNet3+
UNet3+ takes a full-scale approach by introducing extensive skip connections that link each decoder stage with all encoder stages, rather than just the corresponding level. This multi-scale feature fusion integrates high-level semantic information with low-level spatial details at every decoder level, enhancing both fine-grained segmentation and global context awareness.

 > [!Note]
 > For more information, see the corresponding arxiv paper.
 > - [UNet arxiv paper](https://arxiv.org/abs/1505.04597)
 > - [UNet++ arxiv paper](https://arxiv.org/abs/1807.10165)
 > - [UNet3+ arxiv paper](https://arxiv.org/abs/2004.08790)


*****


# 📋Model Comparison

| Model Architecture | Backbone  | Number of Parameters |
|--------------------|-----------|----------------------|
| UNet              | -         | 31.04M              |
| ResUNet18         | ResNet18  | 20.78M              |
| ResUNet34         | ResNet34  | 30.89M              |
| ResUNet50         | ResNet50  | 40.90M              |
| ResUNet101        | ResNet101 | 59.89M              |
| ResUNet152        | ResNet152 | 75.54M              |

*****


# 🔨Getting Started

## 1. Check the Running Environment

## 2. Installation and Dependencies

## 3. Downloading Pre-Trained ResNet Weights

## 4. 


*****


# 📁Project Structure

```bash
UNet-Segmentation
├── assets/           # 
│   └── unet.png
├── dataset/          # 
│   └── dataset.py
├── model/            # 
│   ├── modules.py
│   ├── resnet.py
│   ├── unet.py
│   ├── unet2+.py
│   ├── unet3+.py
│   └── utils.py
├── train/            # 
│   ├── loss.py
│   └── trainer.py
├── README.md         # 
└── requirements.txt  # 

```

