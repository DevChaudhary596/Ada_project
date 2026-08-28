# 🩺 MedVision-Agent

> **Attention-Guided Medical Image Segmentation with an Autonomous Clinical AI Co-Pilot**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MedVision-Agent is an end-to-end Computer Vision and Agentic AI system designed for dermatological lesion segmentation, automated ABCD biomarker extraction, epistemic uncertainty quantification, and clinical triage reporting.

---

## 🌟 Key Features

* 🧠 **Attention U-Net Architecture:** Features Attention Gates ($\alpha$-weighting) to filter out hair artifacts, vignette shadows, and background skin texture.
* 🔬 **Morphological Biomarkers (Stolz ABCD Rule):** Computes **Asymmetry**, **Border Irregularity**, **Color Variegation**, **Diameter**, and **Total Dermatoscopy Score (TDS)**.
* 🛡️ **Epistemic Uncertainty Estimation:** Uses **Monte Carlo Dropout** to visualize boundary ambiguity and compute confidence scores.
* 🤖 **Autonomous Clinical Co-Pilot (ReAct Pattern):** Orchestrates multi-step reasoning, tool execution, and synthesizes structured medical diagnostic reports.
* 🎨 **Interactive Web Dashboard:** Modern Streamlit UI with **Light & Dark Theme** toggles, real-time overlays, and exportable Markdown/PDF reports.
* ⚡ **Ultra-Lightweight & CPU-Ready:** Runs smoothly on standard laptops (CPU) as well as NVIDIA GPUs.

---

## 🏗️ Architecture

```
Input Scan (256x256x3)
      │
      ▼
PyTorch Attention U-Net Engine ──► Attention Gate Map & Binary Mask
      │
      ▼
Autonomous Clinical Co-Pilot (ReAct Loop)
 ├── Tool 1: SegmentationTool
 ├── Tool 2: BiomarkerTool (Asymmetry, Border, Color, Diameter)
 ├── Tool 3: UncertaintyTool (Monte Carlo Dropout Variance)
 └── Tool 4: GuidelinesTool (Stolz ABCD Dermatology Protocol)
      │
      ▼
Interactive Web UI & Downloadable Clinical Diagnostic Report
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/MedVision-Agent.git
cd MedVision-Agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
# Option A: Command line
streamlit run app.py

# Option B: Double-click run.bat on Windows
```
Open your browser at `http://localhost:8501`.

---

## 📖 Research References
1. **U-Net:** *Convolutional Networks for Biomedical Image Segmentation* (Ronneberger et al., MICCAI 2015) — [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
2. **Attention U-Net:** *Learning Where to Look for the Pancreas* (Oktay et al., MIDL 2018) — [arXiv:1804.03999](https://arxiv.org/abs/1804.03999)
3. **ABCD Rule of Dermatoscopy:** (Stolz et al., Eur J Dermatol)

---

## 📐 Mathematical Formulation

### 1. Attention Gate Spatial Weighting:
$$\alpha = \sigma(\psi^T(\text{ReLU}(W_x x + W_g g + b_g)) + b_\psi)$$

### 2. Dice Similarity Coefficient (DSC):
$$\text{Dice}(Y, \hat{Y}) = \frac{2 |Y \cap \hat{Y}|}{|Y| + |\hat{Y}|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

### 3. Total Dermatoscopy Score (TDS):
$$\text{TDS} = 1.3 \times A + 0.1 \times B + 0.5 \times C + 0.5 \times D$$

---

Model Weights: Due to the large size of the trained PyTorch model (best_attention_unet.pth), the model weights are hosted separately rather than being included in this repository. Please download the model from the link provided below and place it in the root project directory to run MedVision Agent successfully.
https://drive.google.com/file/d/1TdurudJ2T7RkQ9g5vTmZwbZERBdKlsYF/view?usp=sharing
---

## 📄 License
This project is licensed under the MIT License.
