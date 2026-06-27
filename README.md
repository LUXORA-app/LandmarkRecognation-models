# 🤖 LUXORA — ML Models

> ConvNeXt-based monument recognition model for ancient Egyptian landmarks in Luxor

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-84%25-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

---

## 📌 Overview

LUXORA's ML pipeline is a fine-tuned **ConvNeXt-Tiny** model trained to recognize 4 major ancient Egyptian landmarks in Luxor from photos. The model is designed for mobile deployment and is integrated into the LUXORA app via the Laravel backend.

---

## 🏛️ Supported Landmarks

| # | Landmark | Images |
|---|---|---|
| 1 | Colossi of Memnon | 156 |
| 2 | Hatshepsut Temple | 156 |
| 3 | Karnak Precinct | 141 |
| 4 | Luxor Temple | 165 |

---

## 📦 Dataset

A specialized dataset was **manually compiled** from high-quality images of the most prominent archaeological sites in Luxor.

- **Total Images:** 618 images across 4 classes (~141 avg per class)
- **Split:** 80% Training / 20% Validation
- **Source:** Custom curated dataset

> 📥 https://drive.google.com/drive/folders/10Qgjgqs1nBMobn4t-vfqh1v8z4VfgkVi?usp=drive_link

### Preprocessing & Augmentation

| Step | Details |
|---|---|
| Image Resizing | 224×224 pixels (ConvNeXt requirement) |
| Normalization | Pixel values scaled to [0, 1] |
| Augmentation | Rotation, Zoom, Horizontal Flip |
| Class Weights | Auto-calculated to handle class imbalance |

---

## 🏗️ Model Architecture

**Base Model: ConvNeXt-Tiny**

ConvNeXtTiny was selected as the final architecture for several reasons:
- Combines the power of CNNs with the efficiency of modern Transformer architectures
- Achieves high accuracy on ImageNet-scale classification tasks
- Excellent computational efficiency — suitable for mobile deployment after conversion

### Fine-Tuning Strategy

```
ConvNeXt-Tiny (pretrained on ImageNet)
├── First 100 layers → Frozen (retains edges, shapes, textures)
└── Remaining layers → Trainable (adapts to Luxor monuments)
    └── Global Average Pooling
        └── Dense Layer + Softmax (4 classes)
```

### Training Configuration

| Parameter | Value |
|---|---|
| Loss Function | Categorical Crossentropy |
| Optimizer | Adam |
| Learning Rate | 5e-6 |

---

## 📊 Model Comparison

| Model | Notes | Performance |
|---|---|---|
| ResNet | Strong classical architecture with residual connections | Good, less than ConvNeXt |
| MobileNet | Lightweight model designed for mobile | Lower accuracy due to small size |
| MobileNet + ConvNeXt (Hybrid) | Combined architecture | Significant improvement, higher complexity |
| **ConvNeXt-Tiny** ✅ | **Final selected model** | **Best accuracy, least complexity** |

---

## 📈 Results & Evaluation

### Overall Performance

| Metric | Score |
|---|---|
| ✅ Overall Accuracy | **84%** |
| 📉 Validation Loss | **~0.47** |

### Per-Class Performance

| Landmark | Precision | Recall | F1-Score | Notes |
|---|---|---|---|---|
| Hatshepsut Temple | 0.96 | 0.96 | **0.96** | Superior — clear visual features |
| Colossi of Memnon | 0.88 | 0.86 | **0.87** | Stable results |
| Karnak Precinct | 0.80 | 0.82 | **0.81** | — |
| Luxor Temple | 0.78 | 0.80 | **0.79** | Architectural similarity to Karnak |
| **Weighted Average** | **0.86** | **0.84** | **0.85** | — |

### Learning Curves

- **Validation Loss** decreased gradually and stabilized at ~0.47
- **No significant gap** between Training Loss and Validation Loss → no overfitting
- **Accuracy** increased steadily, reaching **84%** on the validation set

---

## 🚀 Getting Started

### Prerequisites

- Python `>= 3.10`
- pip or conda

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/luxora-app/ml-models.git
cd ml-models

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the inference API
cd api
python app.py
```

---

## 📁 Project Structure

```
ml-models/
├── data/
│   ├── raw/                  # Raw collected images
│   ├── processed/            # Preprocessed dataset
│   └── augmented/            # Augmented training data
├── models/
│   ├── training/             # Training scripts
│   ├── evaluation/           # Evaluation & metrics
│   └── saved/                # Saved model weights (.h5)
├── notebooks/
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   ├── training.ipynb        # Model training
│   └── evaluation.ipynb      # Results & confusion matrix
├── api/
│   ├── app.py                # Inference API
│   └── utils.py              # Helper functions
├── requirements.txt
└── README.md
```

---

## 👩‍💻 ML Team

| Name | Role |
|---|---|
| Khaled | ML Engineer |
| Mostafa Ezzat | ML Engineer |
| Salma | ML Engineer |

---

## 🔗 Related Repositories

- [📱 Mobile App](../mobile-app) — Flutter Application
- [⚙️ Backend](../backend) — Laravel REST API
- [📄 Docs](../docs) — Full Documentation
