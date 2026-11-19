# 🩺 MedVision-Agent (HAM10000 Multi-Task Edition)

> **HAM10000 Attention-Guided Medical Image Segmentation, 7-Class Diagnostic Classification & Autonomous Clinical AI Co-Pilot**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HAM10000](https://img.shields.io/badge/Dataset-HAM10000-10B981.svg)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MedVision-Agent is an end-to-end Computer Vision and Agentic AI system trained on the **HAM10000 (Human Against Machine with 10000 training images)** dataset and **Tschandl Ground-Truth Segmentation Masks** for simultaneous dermatological lesion segmentation, 7-class diagnostic classification (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`), automated Stolz ABCD biomarker extraction, epistemic uncertainty quantification, and clinical triage reporting.

---

## 🌟 Key Features

* 🧠 **Multi-Task Attention U-Net (HAM10000 Trained):** Features Attention Gates ($\alpha$-weighting) to suppress hair artifacts and vignetting while simultaneously outputting **pixel-level segmentation masks** and **7-class HAM10000 diagnostic predictions**.
* 🧬 **HAM10000 7-Class Taxonomy:** Classifies dermoscopy scans into `akiec` (Actinic Keratoses / Bowen's Disease), `bcc` (Basal Cell Carcinoma), `bkl` (Benign Keratosis), `df` (Dermatofibroma), `mel` (Melanoma), `nv` (Melanocytic Nevus), and `vasc` (Vascular Lesion) with Top-3 differential probabilities.
* 📦 **Automated HAM10000 Dataset Loader (`dataset.py`):** Automatically fetches official `HAM10000_metadata.csv`, all 10,015 Tschandl lesion segmentation masks (`HAM10000_segmentations_lesion_tschandl`), and stratified or full dermoscopy scan archives from HuggingFace Hub, or loads from a local `data/HAM10000` directory.
* 🏋️ **End-to-End Training & Ablation Pipeline (`train.py`):** Trains the Multi-Task Attention U-Net with a Hybrid Dice + BCE + Class-Weighted Label-Smoothed Cross-Entropy Loss (`MultiTaskHAM10000Loss`), runs a scientific ablation study against Baseline U-Net, and exports `best_attention_unet.pth` + `training_history.json`.
* 🔬 **Morphological Biomarkers (Stolz ABCD Rule):** Computes **Asymmetry**, **Border Irregularity**, **Color Variegation**, **Diameter**, and **Total Dermatoscopy Score (TDS)**.
* 🛡️ **Epistemic Uncertainty Estimation:** Uses **Monte Carlo Dropout** to visualize boundary ambiguity and compute confidence scores.
* 🤖 **Autonomous Clinical Co-Pilot (5-Tool ReAct Pattern):** Orchestrates multi-step reasoning across `SegmentationTool`, `HAM10000ClassificationTool`, `BiomarkerTool`, `UncertaintyTool`, and `GuidelinesTool`.

---

## 🏗️ Architecture

```
HAM10000 Dermoscopy Scan (256x256x3) + Clinical Metadata
      │
      ▼
Multi-Task PyTorch Attention U-Net Engine
 ├── Decoder Path ──► Attention Gate Map (α) & Binary Lesion Mask
 └── Multi-Scale Pooled Head ──► 7-Class HAM10000 Softmax Probabilities
      │
      ▼
Autonomous Clinical Co-Pilot (ReAct Loop)
 ├── Tool 1: SegmentationTool (Attention U-Net Boundary Localization)
 ├── Tool 2: HAM10000ClassificationTool (7-Class Differential Diagnosis)
 ├── Tool 3: BiomarkerTool (Asymmetry, Border, Color, Diameter, TDS)
 ├── Tool 4: UncertaintyTool (Monte Carlo Dropout Variance)
 └── Tool 5: GuidelinesTool (Integrated HAM10000 + Stolz ABCD Triage Protocol)
      │
      ▼
Interactive Web UI, Training Dashboard & Hospital-Grade PDF Report
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train on the HAM10000 Dataset
```bash
# Fast stratified training pass (downloads metadata, 10,015 masks, and balanced 7-class subset)
python train.py --epochs 12 --batch-size 8 --samples-per-class 25

# Full 10,015 image archive download & training (~2.6 GB)
python train.py --epochs 30 --batch-size 16 --full-download

# Or train using an existing local HAM10000 folder
python train.py --data-dir /path/to/HAM10000 --epochs 25
```

### 3. Run the Application
```bash
# Option A: Command line
streamlit run app.py

# Option B: Double-click run.bat on Windows
```
Open your browser at `http://localhost:8501`. You can also trigger training or fine-tuning directly from **Tab 5 (🏋️ HAM10000 Training & Benchmarks)** inside the web interface.

---

## 📖 Research References
1. **HAM10000 Dataset:** *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions* (Tschandl et al., *Scientific Data* 2018)
2. **U-Net:** *Convolutional Networks for Biomedical Image Segmentation* (Ronneberger et al., *MICCAI* 2015) — [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
3. **Attention U-Net:** *Learning Where to Look for the Pancreas* (Oktay et al., *MIDL* 2018) — [arXiv:1804.03999](https://arxiv.org/abs/1804.03999)
4. **ABCD Rule of Dermatoscopy:** (Stolz et al., *Eur J Dermatol*)
