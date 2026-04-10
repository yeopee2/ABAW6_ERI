# ABAW6_ERI

# Human Reaction Intensity Estimation with MTL-DAN

## Overview

This repository contains the code for our **Emotional Reaction Intensity (ERI)** project.
The goal is to predict the intensity of 7 emotional reactions from facial image sequences:

- Adoration
- Amusement
- Anxiety
- Disgust
- Empathic-Pain
- Fear
- Surprise

Our pipeline is:

1. Extract features with **MTL-DAN**
2. Train a sequence model such as **LSTM**
3. Ensemble multiple models
4. Generate final prediction files

---

## Data

This repository does **not** include the dataset.

Please download the data from the **official source** used in the **ABAW 5 ERI challenge**.

Required data:
- **Hume-Reaction** dataset
- Official train/validation/test split files if available
- Cropped face frames or your own prepared face image sequences

Please follow the official dataset license and access rules.

---

## Repository Structure

```text
.
├─ save_features.py
├─ train_sequence_mtl.py
├─ train_by_features.py
├─ ensemble.py
├─ test.py
├─ calculate_pcc_csv.py
├─ networks/
└─ utils/
