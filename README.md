# 🧘 Lightweight Method for Yoga Posture Recognition

> **Published research paper** · Institute of Computing – Federal University of Amazonas (UFAM)

A lightweight neural network for automatic yoga posture recognition from images, leveraging skeletal keypoints extracted with MediaPipe. Achieves **90.31% accuracy across 82 yoga classes** with only **85,582 parameters** — outperforming MobileNetV2 while being 27x smaller.

---

## 🏆 Results at a Glance

| Model | Parameters | Model Size | Accuracy | F1-Score |
|---|---|---|---|---|
| **Ours (Proposed)** | **85,582** | **1.10 MB** | **90.31%** | **0.8905** |
| MobileNetV2 | 2,363,026 | 27.4 MB | 78.67% | 0.7482 |
| Swain et al. (2022) | 57,498 | 716 KB | 86.31% | 0.8513 |

Our model outperforms MobileNetV2 by **+11.64%** in accuracy while using **27x fewer parameters**.

---

## 📄 Paper

**Lightweight Method for Yoga Posture Recognition: Contributions to Well-Being and Quality of Life**  
Caio C. M. Antunes, Rafael C. Carvalho, Bernardo B. Gatto, Juan G. Colonna  
Institute of Computing – Federal University of Amazonas (UFAM)  
📎 [Read the paper](https://doi.org/10.5753/sbcas.2025.6971)

---

## 🏗️ Architecture

A hybrid architecture combining:
- **Multi-Head Attention** (4 heads, key dim 16) — captures global relationships between keypoints
- **1D Convolutional Layers** (80 → 48 → 16 filters) — extracts local spatial features
- **Transformer Encoder** (4 attention heads) — models sequential dependencies
- **Dense Layers** (88 → 82 neurons) — final classification

Input: `R^(33×4)` keypoint matrix extracted by MediaPipe Pose  
Output: probability distribution over **82 yoga classes**

---

## 📁 Repository Structure

```
yoga_class_method/
├── dataset_gens/          # Scripts to generate keypoint datasets from Yoga-82
│   ├── img_data_cl.py     # Dataset cleaning with MediaPipe
│   ├── img_data_redist.py # Dataset redistribution
│   ├── img_data_aug.py    # Horizontal flip augmentation
│   ├── kp_ds_gen.py       # Keypoint extraction
│   └── kp_data_aug.py     # Gaussian noise + keypoint fusion balancing
├── experiments/           # Jupyter notebooks for training, testing, and validation
└── model_tester.py        # Latency comparison script
```

---

## 🗄️ Datasets

Three dataset versions were evaluated:

| Dataset | Description |
|---|---|
| `kp_ds_not_aug_1` | Keypoints from cleaned images, no augmentation |
| `kp_ds_aug_2` | + Horizontal flip (doubles samples per class) |
| `kp_ds_aug_balan_3` | + Gaussian noise & keypoint fusion for class balancing |

📥 Download the datasets [here](https://drive.google.com/drive/folders/1J22NMrp7-ASANnqbkdPJ8ay9WPHqV_VG?usp=sharing) 
Original image dataset: [Yoga-82](https://sites.google.com/view/yoga-82/home) (Verma et al., 2020)

---

## ⚙️ How to Reproduce

### Prerequisites
```bash
pip install tensorflow mediapipe numpy
```

### Step 1 — Generate the keypoint dataset

```bash
# Clean the image dataset
python dataset_gens/img_data_cl.py

# Redistribute samples
python dataset_gens/img_data_redist.py

# Extract keypoints (rename output to kp_ds_not_aug_1)
python dataset_gens/kp_ds_gen.py
```

### Step 2 — Run the experiments

Open and execute the notebooks inside `experiments/` in order.

### Step 3 — Test inference latency

```bash
python model_tester.py
```

---

## 🔬 Data Augmentation Strategies

**Horizontal Flipping** — mirrors images to simulate camera/person rotation, doubling the number of samples per class.

**Gaussian Noise** — adds subtle spatial perturbations (µ=0, σ=0.01) to keypoint coordinates, simulating natural variations in poses.

**Keypoint Fusion** — interpolates two samples from the same class using a weighted average:

```
Kc = α·K1 + (1−α)·K2,  α ∈ [0.3, 0.7]
```

---

## 💡 Key Findings

- **Horizontal flipping alone** yielded the best results (90.31%), outperforming more complex balancing strategies
- **Keypoint-based models** consistently outperform image-based models on this dataset
- The model is suitable for **resource-constrained environments** such as smartphones (1.10 MB, ~1.6s latency on CPU)

---

## 🏛️ Acknowledgments

This work was funded by Samsung through the Informatics Law for the Western Amazon (Federal Law nº 8.387/1991), CAPES-PROEX (Funding Code 001), and FAPEAM through the PDPG project. R&D project 001/2020, signed with UFAM and FAEPI.

---

## 📚 Citation

```bibtex
@article{antunes2025yoga,
  title={Lightweight Method for Yoga Posture Recognition: Contributions to Well-Being and Quality of Life},
  author={Antunes, Caio C. M. and Carvalho, Rafael C. and Gatto, Bernardo B. and Colonna, Juan G.},
  year={2025}
}
```
