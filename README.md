

# **GMamba**

**Enabling True Global Perception in State Space Models for Visual Tasks**

Official repository of the ICLR 2026 paper:


***Enabling True Global Perception in State Space Models for Visual Tasks***

🔗 **Paper (OpenReview):** [https://openreview.net/forum?id=96p1eBUVaN](https://openreview.net/forum?id=96p1eBUVaN)

---

## 🌟 Key Contributions

<p align="center">
  <img width="80%" src="https://github.com/user-attachments/assets/98422b2f-faae-46a9-a7c4-e6302d060837" alt="GMamba overview" />
</p>

* **A Formal Definition of Global Perception in Vision**
  We introduce the first rigorous mathematical definition of *visual global perception*, formulated via the gradient behavior of the Frobenius norm.
  This provides a principled criterion to distinguish truly global models from locally biased approximations.

* **GSSM: Global State Space Model**
  We propose **GSSM**, a frequency-aware state space module using *Discrete Fourier Transform (DFT)-based pre-modulation* to overcome the exponential decay inherent in recursive SSMs.
  This enables **stable long-range dependency modeling** in vision tasks.

<p align="center">
  <img width="80%" src="https://github.com/user-attachments/assets/d06be703-204c-4398-9703-9838d8c14f28" alt="GSSM module" />
</p>

* **GMamba Architecture**
  GMamba is a **plug-and-play global perception module** with **linear-logarithmic complexity**
  [
  \mathcal{O}(N \log N)
  ]
  It can be seamlessly integrated into CNN- or Transformer-based architectures without redesigning the backbone.

* **Consistent Performance Gains Across Tasks**
  GMamba demonstrates strong and consistent improvements over existing attention-based and SSM-based global modeling methods on **semantic segmentation**, **object detection**, and **instance segmentation**.

---

## 🚀 Architecture Overview

**Figure 1** illustrates the overall GMamba architecture and internal design of **GSSM**.

* **FEM (Frequency Encoding Module)**: Maps spatial features to the frequency domain to reveal global spectral structures.
* **FGMM (Frequency-Guided Modulation Mechanism)**: Modulates the state transition dynamics using frequency responses to prevent information vanishing.

This design ensures **true global perception** while maintaining **computational efficiency**.

---

## 📊 Experimental Results

### 1. Remote Sensing Semantic Segmentation

**Vaihingen & Potsdam Datasets**

When integrated into UNet-style backbones, GMamba improves robustness in complex land-cover scenarios, particularly for large and fragmented objects.

| Backbone   | mIoU (Baseline) | **mIoU (+GMamba)** | **Gain** |
| ---------- | --------------- | ------------------ | -------- |
| ResNet34   | 81.65%          | **84.74%**         | +3.09%   |
| Swin-T     | 82.44%          | **84.83%**         | +2.39%   |
| ConvNeXt-S | 83.11%          | **86.00%**         | +2.89%   |

---

### 2. Object Detection

**MS COCO**

When integrated into Faster R-CNN (ResNet-50 backbone), GMamba consistently improves performance, especially for **large objects**.

| Method       | AP       | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
| ------------ | -------- | -------------- | -------------- | -------------- |
| Baseline     | 37.2     | 21.5           | 40.4           | 48.0           |
| **+ GMamba** | **38.5** | **22.1**       | **42.2**       | **49.9**       |

---

## 📑 **Citation**

```bibtex
@inproceedings{
hui2026enabling,
title={Enabling True Global Perception in State Space Models for Visual Tasks},
author={Jie Hui and Zhenxiang Zhang and Wenyu Mi and Jianji Wang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=96p1eBUVaN}
}
```

---

