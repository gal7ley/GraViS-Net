# SynergyNet: Dual-Stream Network with Topology-Awareness for Skin Lesion Diagnosis

![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-%23EE4C2C.svg?style=flat&logo=pytorch) ![Python](https://img.shields.io/badge/Python-3.8-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

> **Note:** This repository contains the official PyTorch implementation of **SynergyNet**. This code is currently anonymized for **IJCAI review**.

## 📖 Introduction

SynergyNet is a novel dual-stream framework designed for long-tailed skin lesion classification. It integrates appearance features (RGB) with topological structural features derived from the **Sigmoid-Weighted Image Visibility Graph (SW-IVG)** to enhance diagnostic accuracy and robustness.

**Key Features:**
*   **Dual-Stream Architecture:** Combines a strong ResNet50 backbone for visual details (RGB) and a lightweight ResNet18 backbone for topological structures (SW-IVG).
*   **SW-IVG Topology:** Utilizes a novel visibility graph construction method to capture structural patterns that are robust to noise and artifacts.
*   **Offline Preprocessing:** Implements an $O(N^2)$ visibility graph construction offline using Numba acceleration, converting complex topology calculation into efficient O(1) data loading during training.
*   **Geometric Alignment:** Ensures strict spatial consistency between RGB images and topological maps during data augmentation using `Albumentations`.
*   **Metadata Integration:** Incorporates clinical metadata (e.g., age, anatomical site) via a **FiLM Modulator** for enhanced performance on ISIC 2019 and PAD-UFES-20 datasets.
*   **Long-Tail Handling:** Employs Deferred Re-balancing (DRW) strategies and LDAM Loss to mitigate class imbalance.

## 📂 Project Structure

```text
SynergyNet/
├── checkpoints/        # Directory for saving model weights and logs
├── models_v5.py        # SynergyNet architecture definition (Dual-Stream + FiLM)
├── utils_swivg.py      # Core SW-IVG algorithm (Numba accelerated)
├── utils_dataset.py    # Dataset loader with geometric alignment logic
├── utils_loss.py       # LDAM Loss implementation
├── main_synergy.py     # Main training and evaluation script
├── precompute_swivg.py # Script for offline topological map generation
├── requirements.txt    # Python dependencies
└── README.md

conda create -n synergy python=3.8
conda activate synergy
Install dependencies:
pip install -r requirements.txt
🚀 Data Preparation
1. Download Datasets
Please download the datasets from their official sources.
ISIC 2018 (Task 3: Lesion Diagnosis):
Download Link: https://challenge.isic-archive.com/data/
Instructions: Select Year 2018 -> Download Task 3: Training Input and Task 3: Training Ground Truth.
ISIC 2019:
Download Link: https://challenge.isic-archive.com/data/
Instructions: Select Year 2019 -> Download Training Input and Training Ground Truth.
PAD-UFES-20:
Available on Mendeley Data. Please download the dataset containing smartphone images and metadata.
2. Offline SW-IVG Computation
To avoid high computational overhead during training, we pre-compute the topological maps and cache them as .npy files.
Run the following script to generate the cache:

# Example for ISIC 2018
python precompute_swivg.py \
  --data_dir /path/to/ISIC2018_Images \
  --save_dir ./cache_swivg_2018
Note: This script uses Numba for parallel acceleration. On a standard 16-core CPU, processing the ISIC 2018 dataset takes approximately 20 minutes.
Output: The script will generate a directory containing .npy files corresponding to each image ID.
⚡ Usage
1. Training from Scratch (ISIC 2018 / 2019)
Train the model using the Deferred Re-balancing (DRW) strategy with LDAM Loss.

python main_synergy.py \
  --model_name SynergyNet_ISIC \
  --base_dir /path/to/dataset \
  --cache_dir ./cache_swivg_2018 \
  --num_classes 7 \
  --batch_size 32 \
  --epochs 80 \
  --lr_rgb 2e-5 --lr_new 2e-4
2. Transfer Learning (PAD-UFES-20)
Fine-tune the model (pre-trained on ISIC 2019) for the small-scale PAD-UFES-20 dataset. This mode uses Label Smoothing instead of LDAM Loss.
python main_synergy.py \
  --model_name SynergyNet_PAD \
  --base_dir /path/to/PAD_dataset \
  --pretrained_path ./checkpoints/isic_2019_best.pth \
  --num_classes 6 \
  --lr_rgb 2e-5 --lr_new 2e-4
📊 Results
The model performance (averaged over 5-fold cross-validation) is summarized below:
Dataset	Accuracy	Balanced Acc.	AUC
ISIC 2018	89.4%	81.2%	0.975
ISIC 2019	87.1%	76.5%	0.962
PAD-UFES-20	85.3%	82.0%	0.941
