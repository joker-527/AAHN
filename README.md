## Introduction

This is the source code of the paper <strong>Asymmetric Adaptive Heterogeneous Network for Multi-Modality Medical Image Segmentation</strong>

The codebase is based from [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1)

![Network Architecture](/images/net.png "Network Architecture")


## Abstract

Existing studies of multi-modality medical image segmentation tend to aggregate all modalities without discrimination and employ multiple symmetric encoders or decoders for feature extraction and fusion. They often overlook the different contributions to visual representation and intelligent decisions among multi-modality images. Motivated by this discovery, this paper proposes an asymmetric adaptive heterogeneous network for multi-modality image feature extraction with modality discrimination and adaptive fusion. For feature extraction, it uses a heterogeneous two-stream asymmetric feature-bridging network to extract complementary features from auxiliary multi-modality and leading single-modality images, respectively. For feature adaptive fusion, the proposed Transformer-CNN Feature Alignment and Fusion (T-CFAF) module enhances the leading single-modality information, and the Cross-Modality Heterogeneous Graph Fusion (CMHGF) module further fuses multi-modality features at a high-level semantic layer adaptively. Comparative evaluation with ten segmentation models on six datasets demonstrates significant efficiency gains as well as highly competitive segmentation accuracy. 

## Getting Started

- Installation
  <br /> Follow [nnUNet's](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) installation process.
- For training
  <br /> `nnUNet_train 2d nnUNetTrainer_TransUNet_Graph_TranFuse_v2_flair Task*** all `
- For testing
  <br /> ` nnUNet_predict -i /inputDir -o /outputDir -m 2d -t Task*** -f all `

## Result

![Result of Hecktor21 Dataset](/images/hecktor21_result.png "Result of Hecktor21 Dataset")
![Result of Prostate158 Dataset](/images/prostate158_result.png "Result of Prostate158 Dataset")
![Result of BraTS2019 Dataset](/images/brats19_result.png "Result of BraTS2019 Dataset")
![Result of BraTS2023 Dataset](/images/brats23_result.png "Result of BraTS2023 Dataset")
![Result of CHAOS Dataset](/images/chaos_result.png "Result of CHAOS Dataset")
![Result of BraTS2024 Dataset](/images/brats24_result.png "Result of BraTS2024 Dataset")
