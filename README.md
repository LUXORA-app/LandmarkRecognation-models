# 🤖 LUXORA — ML Models

> ConvNeXt-based monument recognition model and YOLOv26s-based temple scene detection for ancient Egyptian landmarks in Luxor

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

## 📜 Scene Recognition Model for Luxor Temple

### Background & Objectives
The Luxor Temple stands as a monumental archive of ancient Egyptian history, containing dense, overlapping scenes of distinct deities and historical figures. Standard image classification models are insufficient to localize multiple overlapping subjects within a single frame. Therefore, a custom object detection pipeline utilizing the **YOLOv26s** architecture was developed to recognize 11 distinct motif classes in real-time.

### Dataset & Preprocessing
A specialized dataset of 510 high-quality annotated images was compiled and partitioned into a 70% Training (357 images), 20% Validation (102 images), and 10% Testing (51 images) split.

To artificially expand the training set to 1,071 instances and prevent overfitting on underrepresented classes, robust augmentations were applied:
- **Spatial:** Random rotation between -15° and +15°.
- **Color:** Grayscale (15% probability), Hue (±15°), Saturation (±25%), and Brightness (±15%).
- **Degradation:** Blur (up to 2px) and noise addition (up to 1%) to mimic poor lighting and sensor noise in temple spaces.
- **Mosaic Augmentation (0.5):** Stitches four images into a grid to force the model to recognize smaller-scale shapes. Mosaic generation was disabled (`close_mosaic=15`) in the final 15 epochs to recalibrate bounding box precision.

### Technical Architecture
Built on the small variant of the **YOLOv26** framework (`yolo26s.pt`):
- **Backbone & Neck:** High-efficiency convolutional blocks and a Path Aggregation Network (PANet) designed for edge-first execution.
- **Head (NMS-Free):** Features a natively decoupled detection head that removes the traditional Non-Maximum Suppression computational bottleneck, heavily accelerating edge CPU inference speeds.
- **Optimization:** Utilizes Binary Cross-Entropy (BCE) for classification loss and Complete IoU (CIoU) for bounding box regression, dropping the DFL module for maximum hardware efficiency. Pre-trained COCO weights were used for transfer learning.

### Results & Evaluation
The YOLOv26s model stabilized successfully without overfitting (using `patience=50` for early stopping). 

| Metric | Score |
|---|---|
| **mAP@50** | **97.7%** |
| **mAP@50-95** | **80.7%** |
| **Overall Precision** | **95.7%** |
| **Overall Recall** | **95.7%** |

#### Class-Specific Performance (Test Set)

| Class | Instances | Precision | Recall | mAP@50 |
|---|---|---|---|---|
| asiatics | 17 | 0.997 | 1.000 | 0.995 |
| africans | 12 | 1.000 | 1.000 | 0.995 |
| alexander | 14 | 0.986 | 1.000 | 0.995 |
| amun | 27 | 0.928 | 0.948 | 0.968 |
| horus | 9 | 1.000 | 0.946 | 0.995 |
| monkeys | 2 | 0.915 | 1.000 | 0.995 |
| mut | 4 | 0.918 | 0.750 | 0.945 |
| ramses | 20 | 1.000 | 0.887 | 0.895 |
| son of ramses | 9 | 0.899 | 0.994 | 0.973 |
| uniting scene | 4 | 0.943 | 1.000 | 0.995 |
| werethekau | 3 | 0.941 | 1.000 | 0.995 |

---

## 🚀 Getting Started

### Prerequisites

- Python `>= 3.10`
- pip or conda

### Installation

```bash
# 1. Clone the repository
git clone [https://github.com/luxora-app/ml-models.git](https://github.com/luxora-app/ml-models.git)
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

## 🔗 Related Repositories

- [📱 Mobile App](../mobile-app) — Flutter Application
- [⚙️ Backend](../backend) — Laravel REST API
- [📄 Docs](../docs) — Full Documentation
