# SynergyNet: Dual-Stream Network with Topology-Awareness for Skin Lesion Diagnosis

![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-%23EE4C2C.svg?style=flat&logo=pytorch) ![Python](https://img.shields.io/badge/Python-3.8-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

> **Note:** This repository contains the official PyTorch implementation of **SynergyNet**. This code is currently anonymized for **IJCAI review**.

## 📖 Introduction

SynergyNet is a novel dual-stream framework designed for long-tailed skin lesion classification. It integrates appearance features (RGB) with topological structural features derived from the **Sigmoid-Weighted Image Visibility Graph (SW-IVG)** to enhance diagnostic accuracy and robustness.

**Key Features:**
*   **Dual-Stream Architecture:** Combines a strong ResNet50 backbone for visual details (RGB) and a lightweight ResNet18 backbone for topological structures (SW-IVG).
*   **SW-IVG Topology:** Utilizes a novel visibility graph construction method to capture structural patterns that are robust to noise and artifacts.
*   **Offline Preprocessing:** Implements an O(N^2) visibility graph construction offline using Numba acceleration, converting complex topology calculation into efficient O(1) data loading during training.
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
