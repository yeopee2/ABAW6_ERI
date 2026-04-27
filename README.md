# ABAW6_ERI

# Human Reaction Intensity Estimation with MTL-DAN

## Overview

This repository contains the implementation for an **Emotional Reaction Intensity (ERI)** estimation project based on the ABAW ERI challenge.

The goal of this project is to estimate the intensity of seven emotional reactions from facial image sequences:

- Adoration
- Amusement
- Anxiety
- Disgust
- Empathic Pain
- Fear
- Surprise

This project follows a deep learning-based affective computing pipeline.  
Facial image sequences are first processed using an MTL-DAN-based facial affect representation model. The extracted frame-level representations are then used as sequential input for recurrent neural network models, such as LSTM, GRU, and BiLSTM, to estimate the intensity of emotional reactions.

The overall workflow is:

1. Load facial image sequences and ERI labels
2. Standardize each video into a fixed-length frame sequence
3. Extract multi-task facial affect features using MTL-DAN
4. Train a sequence regression model
5. Evaluate prediction performance using Pearson Correlation Coefficient (PCC)
6. Save prediction results and training curves

---

## Background

Emotional Reaction Intensity (ERI) estimation is a fine-grained affective computing task.  
Unlike simple facial expression classification, ERI aims to estimate how strongly a person experiences multiple emotional reactions.

This project is based on the ABAW ERI task using the Hume-Reaction dataset.  
The original paper reports that the dataset contains approximately 75 hours of video recordings from 2,222 subjects across South Africa and the United States. Each sample is annotated with intensity scores for seven emotional reactions on a scale from 1 to 100.

The main idea of this project is to combine:

- **MTL-DAN** for facial affect feature extraction
- **Sequence models** such as LSTM, GRU, and BiLSTM for temporal modeling
- **PCC/CCC-based loss functions** for regression-based ERI optimization

---

## Model Architecture

The implemented model consists of two main parts.

### 1. MTL-DAN Feature Extractor

The MTL-DAN module extracts facial affect features from each facial frame.

The implementation uses a ResNet-based DAN architecture with cross-attention heads.  
The attention module consists of spatial attention and channel attention blocks. These are used to extract robust facial affect representations from facial images.

The MTL-DAN branch produces three types of affect-related outputs:

- **EXPR**: facial expression representation with 8 dimensions
- **AU**: action unit representation with 12 dimensions
- **VA**: valence-arousal representation with 2 dimensions

These outputs are concatenated into a 22-dimensional feature vector:

```text
8 EXPR features + 12 AU features + 2 VA features = 22-dimensional feature vector
```

The code uses pretrained checkpoints such as:

```text
resnet18_msceleb.pth
affecnet8_epoch5_acc0.6209.pth
```

These pretrained weights are used to initialize the facial representation model.

---

### 2. Sequence Regression Model

After extracting MTL-DAN features, the model uses sequential regression networks to estimate ERI scores.

The sequence models take input in the following shape:

```text
(batch_size, sequence_length, feature_dimension)
```

In this project, the default input structure is:

```text
(batch_size, 12, 22)
```

where:

- `12` is the fixed number of frames per video sequence
- `22` is the concatenated MTL-DAN feature dimension

The implemented sequence models include:

- LSTM
- GRU
- BiLSTM
- LSTM with dropout
- LSTM with additional fully connected layers
- Conv-LSTM
- Transformer encoder

The final output layer uses a sigmoid activation function to predict the seven ERI values.

```text
Output dimension = 7
```

---

## Data Processing

Each video is represented as a sequence of facial image frames.

In the dataset loader, each sample is adjusted to have exactly 12 frames:

- If a video has more than 12 frames, frames from the beginning and end are removed.
- If a video has fewer than 12 frames, the last frame is repeated until the sequence length becomes 12.

This creates a fixed-length input sequence for recurrent models.

The label file contains seven emotional reaction intensity scores:

```text
Adoration, Amusement, Anxiety, Disgust, Empathic-Pain, Fear, Surprise
```

The labels are loaded from the dataset information CSV file and converted into a seven-dimensional tensor.

---

## Setup

This project was implemented using **Python** and **PyTorch**.

The original experiment reported in the related paper was conducted using the following hardware environment:

```text
GPU: 6 × NVIDIA RTX 3090
RAM: 128 GB
CPU: Intel i9-10940X
Framework: PyTorch
```

A CUDA-enabled GPU is recommended for running this project.

### Required Libraries

The main libraries used in the implementation are:

```text
torch
torchvision
torchmetrics
numpy
pandas
scikit-learn
Pillow
matplotlib
tqdm
opencv-python
```

---

## Training and Evaluation Procedure

The project can be reproduced through the following general procedure.

### 1. Prepare Facial Image Sequences

First, prepare cropped facial images for each video sample.

Each video should contain multiple facial frames.  
During loading, the code automatically converts the sequence into 12 frames by trimming or repeating frames.

---

### 2. Load ERI Labels

The dataset loader reads the ERI labels from the data information CSV file.

Each sample has seven target values:

```text
[Adoration, Amusement, Anxiety, Disgust, Empathic-Pain, Fear, Surprise]
```

---

### 3. Extract MTL-DAN Features

The MTL-DAN model extracts affective representations from each frame.

For each frame, the model generates:

```text
EXPR features: 8 dimensions
AU features: 12 dimensions
VA features: 2 dimensions
```

These features are concatenated into a 22-dimensional feature vector.

---

### 4. Train Sequence Regression Model

The extracted features are used as input to a sequence model.

The default sequence input shape is:

```text
(batch_size, 12, 22)
```

The sequence model predicts seven ERI scores.

Implemented sequence models include:

```text
LSTM
GRU
BiLSTM
LSTM_drop
LSTM_fc
Conv_LSTM
BiLSTM_fc
TransformerEncoder
```

---

### 5. Optimize with PCC/CCC-based Loss

The implementation includes multiple correlation-based loss functions:

```text
PCCLoss
Single_PCCLoss
Total_PCCLoss
CCCLoss
Single_CCCLoss
Total_CCCLoss
```

The ERI task is evaluated using Pearson Correlation Coefficient (PCC), so PCC-based optimization is directly related to the final evaluation metric.

---

### 6. Save Results

The code saves:

- prediction CSV files
- label CSV files
- training loss curves
- validation loss curves
- training PCC curves
- validation PCC curves

The prediction CSV format contains:

```text
video name, Adoration, Amusement, Anxiety, Disgust, Empathic-Pain, Fear, Surprise
```

---

## Evaluation Metric

The official evaluation metric is the mean **Pearson Correlation Coefficient (PCC)** across the seven emotional reaction categories.

PCC is calculated for each category:

```text
Adoration
Amusement
Anxiety
Disgust
Empathic Pain
Fear
Surprise
```

The final score is the average PCC across all seven categories.

---

## Reported Results

The paper reports a **mean PCC score of 0.3254** on the official validation set for the ERI estimation challenge.

The reported comparison is:

| Method | Mean PCC |
|---|---:|
| ResNet50-VGGFace2 baseline | 0.2488 |
| ResNet50-FAU baseline | 0.2840 |
| MTL-DAN + LSTM regression head | 0.3254 |

These results suggest that combining multi-task facial affect representations with temporal sequence modeling improves ERI estimation performance compared to the baseline methods.

---

## Project Workflow

```text
Facial video frames
   ↓
Frame sequence standardization
   ↓
MTL-DAN feature extraction
   ↓
22-dimensional frame-level affect representation
   ↓
LSTM / GRU / BiLSTM-based sequence regression
   ↓
Seven-dimensional ERI prediction
   ↓
PCC-based evaluation
   ↓
Prediction CSV and training curve export
```

---

## Reproducibility Notes

To reproduce this project, the following resources are required:

1. Hume-Reaction dataset
2. Facial image sequences
3. ERI label CSV file
4. Pretrained MTL-DAN-related checkpoints
5. CUDA-enabled PyTorch environment

The exact performance may vary depending on:

- face detection and cropping quality
- frame sampling and sequence construction
- pretrained checkpoint availability
- random seed
- GPU environment
- selected sequence model
- selected loss function
- ensemble setting

Because the dataset and pretrained checkpoints are not redistributed in this repository, this repository mainly provides the implementation structure for reproducing and reviewing the ERI estimation pipeline.

---

## Skills Demonstrated

This project demonstrates the following implementation skills:

- Processing in-the-wild facial image sequences
- Building a custom PyTorch dataset and dataloader
- Using pretrained facial representation models
- Implementing attention-based facial affect feature extraction
- Constructing MTL representations from EXPR, AU, and VA outputs
- Training recurrent sequence regression models
- Applying PCC/CCC-based loss functions
- Evaluating regression models with PCC
- Saving prediction results and training curves for review

---

## Citation

This repository is based on the following work:

```bibtex
@article{oh2023human,
  title={Human Reaction Intensity Estimation with Ensemble of Multi-task Networks},
  author={Oh, JiYeon and Kim, Daun and Jeong, Jae-Yeop and Hong, Yeong-Gi and Jeong, Jin-Woo},
  journal={arXiv preprint arXiv:2303.09240},
  year={2023}
}
```

---

## Notes

This repository is intended for research practice and reproducibility review.

The dataset and pretrained checkpoints are not redistributed in this repository.  
Please obtain the required data and model weights through the official source and follow the corresponding license and access policy.
