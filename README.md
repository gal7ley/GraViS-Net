# GraViS-Net: Graph-Visual Synergy with Adaptive Soft-Thresholding for Skin Lesion Diagnosis

## 📖 Introduction

GraViS-Net is a novel dual-stream framework designed for long-tailed skin lesion classification. It integrates appearance features (RGB) with topological structural features derived from the **Adaptive Soft-Thresholded Image Visibility Graph (AST-IVG)** to enhance diagnostic accuracy and robustness.

**Key Features:**
*   **Dual-Stream Architecture:** Combines a strong ResNet50 backbone for visual details (RGB) and a lightweight ResNet18 backbone for topological structures (AST-IVG).
*   **AST-IVG Topology:** Utilizes a novel visibility graph construction method to capture structural patterns that are robust to noise and artifacts.
*   **Offline Preprocessing:** Implements an O(N^2) visibility graph construction offline using Numba acceleration, converting complex topology calculation into efficient O(1) data loading during training.
*   **Geometric Alignment:** Ensures strict spatial consistency between RGB images and topological maps during data augmentation using `Albumentations`.
*   **Metadata Integration:** Incorporates clinical metadata (e.g., age, anatomical site) via a **FiLM Modulator** for enhanced performance on ISIC 2019 and PAD-UFES-20 datasets.
*   **Long-Tail Handling:** Employs Deferred Re-balancing (DRW) strategies and LDAM Loss to mitigate class imbalance.

## 📂 Project Structure

```text
GraViS-Net/
├── checkpoints/        # Directory for saving model weights and logs
├── models.py        # GraViS-Net architecture definition (Dual-Stream + FiLM)
├── SWIVG.py      # Core AST-IVG algorithm (Numba accelerated)
├── main_synergy.py     # Main training and evaluation script
├── requirements.txt    # Python dependencies
└── README.md
```
## 🛠️ Installation

1.  **Create a virtual environment** (Recommended):
    ```bash
    conda create -n GraViS-Net python=3.8
    conda activate GraViS-Net
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Data Preparation

### 1. Download Datasets
Please download the datasets from their official sources.

*   **ISIC 2018 (Task 3: Lesion Diagnosis):**
    *   **Download Link:** [https://challenge.isic-archive.com/data/](https://challenge.isic-archive.com/data/)
    *   **Instructions:** Select Year **2018** -> Download **Task 3: Training Input** and **Task 3: Training Ground Truth**.

*   **ISIC 2019:**
    *   **Download Link:** [https://challenge.isic-archive.com/data/](https://challenge.isic-archive.com/data/)
    *   **Instructions:** Select Year **2019** -> Download **Training Input** and **Training Ground Truth**.

*   **PAD-UFES-20 (Clinical Photographs):**
    *   **Features:** Includes 2,298 images with 21 rich clinical metadata features (history, symptoms, Fitzpatrick skin type).

### 📊 Dataset Statistics
| Dataset | Total Images | Classes | Imbalance Ratio | Modality |
| :--- | :---: | :---: | :---: | :--- |
| ISIC 2018 | 10,015 | 7 | 58.3 | Image Only |
| ISIC 2019 | 25,331 | 8 | 53.9 | Image + Metadata |
| PAD-UFES-20 | 2,298 | 6 | 16.3 | Image + Rich Metadata |

### 2. Offline AST-IVG Computation
To avoid high computational overhead during training, we pre-compute the topological maps and cache them as `.npy` files.

Run the following script to generate the cache:

```bash
# Example for ISIC 2018
python SWIVG.py \
  --data_dir /path/to/ISIC2018_Images \
  --save_dir ./cache_swivg_2018
```

*   **Note:** This script uses `Numba` for parallel acceleration. On a standard 16-core CPU, processing the ISIC 2018 dataset takes approximately 20 minutes.
*   **Output:** The script will generate a directory containing `.npy` files corresponding to each image ID.

## ⚡ Usage

### 1. Training from Scratch (ISIC 2018 / 2019)
Train the model using the Deferred Re-balancing (DRW) strategy with LDAM Loss.

```bash
python main_synergy.py \
  --model_name GraViS-Net_ISIC \
  --base_dir /path/to/dataset \
  --cache_dir ./cache_swivg_2018 \
  --num_classes 7 \
  --batch_size 32 \
  --epochs 80 \
  --lr_rgb 2e-5 --lr_new 2e-4
```

### 2. Transfer Learning (PAD-UFES-20)
Fine-tune the model (pre-trained on ISIC 2019) for the small-scale PAD-UFES-20 dataset. This mode uses Label Smoothing instead of LDAM Loss.

```bash
python main_synergy.py \
  --model_name GraViS-Net_PAD \
  --base_dir /path/to/PAD_dataset \
  --pretrained_path ./checkpoints/isic_2019_best.pth \
  --num_classes 6 \
  --lr_rgb 2e-5 --lr_new 2e-4
```
