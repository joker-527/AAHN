## Introduction

This is the source code of the paper <strong>Asymmetric Adaptive Heterogeneous Network for Multi-Modality Medical Image Segmentation</strong>

The codebase is based from [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1)

![Network Architecture](/net.png "Network Architecture")

## Getting Started

- Installation
  Follow [nnUNet's](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) installation process.
- For training
  `nnUNet_train 2d nnUNetTrainer_TransUNet_Graph_TranFuse_v2_flair Task*** all `
- For testing
  ` nnUNet_predict -i /inputDir -o /outputDir -m 2d -t Task*** -f all `
