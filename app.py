import os
import json
import streamlit as st
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from model import AttentionUNet, UNet
from dataset import (
    HAM10000_CLASSES,
    HAM10000_CLASS_INFO,
    get_default_data_dir,
    build_local_manifest,
    download_ham10000_dataset
)
from engine import SyntheticDermoscopyGenerator, generate_pdf_report, compare_longitudinal_scans
from agent import ClinicalCoPilotAgent, ClinicalChatAgent
from train import run_ham10000_training

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="MedVision-Agent | HAM10000 Clinical AI Co-Pilot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Header
st.sidebar.markdown("## 🩺 MedVision-Agent")
st.sidebar.caption("HAM10000 Multi-Task Attention U-Net & Clinical AI")
st.sidebar.markdown("---")

theme = st.sidebar.selectbox("🎨 UI Theme Style", ["Light Mode (Clinical Clean)", "Dark Mode (Cyber Slate)"])

if "Dark" in theme:
    st.markdown("""
    <style>
        .stApp { background-color: #0B0F19; color: #F1F5F9; }
        .main-header { font-size: 2.2rem; font-weight: 800; color: #38BDF8; margin-bottom: 0.1rem; }
        .sub-header { font-size: 1.05rem; color: #94A3B8; margin-bottom: 1.5rem; }

        [data-testid="stMetricValue"] { color: #38BDF8 !important; font-size: 1.65rem !important; font-weight: 700 !important; }
        [data-testid="stMetricLabel"] { color: #CBD5E1 !important; font-size: 0.9rem !important; font-weight: 600 !important; }
        [data-testid="metric-container"] {
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
            padding: 12px !important;
        }

        .stDownloadButton button, .stButton button {
            background-color: #0284C7 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            padding: 10px 20px !important;
            transition: all 0.2s ease !important;
        }
        .stDownloadButton button:hover, .stButton button:hover {
            background-color: #0369A1 !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            background-color: #1E293B !important;
            padding: 6px !important;
            border-radius: 10px !important;
            gap: 6px !important;
            border: 1px solid #334155 !important;
            margin-bottom: 18px !important;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            border: none !important;
        }
        .stTabs [data-baseweb="tab"] * {
            color: #94A3B8 !important;
            font-weight: 700 !important;
            font-size: 0.92rem !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0284C7 !important;
        }
        .stTabs [aria-selected="true"] * {
            color: #FFFFFF !important;
            font-weight: 800 !important;
        }

        .agent-thought { background-color: #064E3B; border-left: 4px solid #10B981; padding: 12px 16px; margin-bottom: 10px; border-radius: 6px; color: #D1FAE5; }
        .agent-action { background-color: #0C4A6E; border-left: 4px solid #0EA5E9; padding: 12px 16px; margin-bottom: 10px; border-radius: 6px; color: #E0F2FE; }
        .chat-bubble-user { background-color: #1E293B; border: 1px solid #475569; padding: 12px 16px; border-radius: 10px; margin-bottom: 10px; color: #F8FAFC; }
        .chat-bubble-agent { background-color: #064E3B; border: 1px solid #059669; padding: 12px 16px; border-radius: 10px; margin-bottom: 10px; color: #ECFDF5; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp { background-color: #F8FAFC; color: #0F172A; }
        .main-header { font-size: 2.2rem; font-weight: 800; color: #0F766E; margin-bottom: 0.1rem; }
        .sub-header { font-size: 1.05rem; color: #475569; margin-bottom: 1.5rem; }

        [data-testid="stMetricValue"] { color: #0F766E !important; font-size: 1.65rem !important; font-weight: 700 !important; }
        [data-testid="stMetricLabel"] { color: #334155 !important; font-size: 0.9rem !important; font-weight: 600 !important; }
        [data-testid="metric-container"] {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
            padding: 12px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
        }

        .stDownloadButton button, .stButton button {
            background-color: #0F766E !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            padding: 10px 20px !important;
            transition: all 0.2s ease !important;
        }
        .stDownloadButton button:hover, .stButton button:hover {
            background-color: #115E59 !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(15, 118, 110, 0.3) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            background-color: #E2E8F0 !important;
            padding: 6px !important;
            border-radius: 10px !important;
            gap: 6px !important;
            border: 1px solid #CBD5E1 !important;
            margin-bottom: 18px !important;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            border: none !important;
        }
        .stTabs [data-baseweb="tab"] * {
            color: #1E293B !important;
            font-weight: 700 !important;
            font-size: 0.92rem !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0F766E !important;
        }
        .stTabs [aria-selected="true"] * {
            color: #FFFFFF !important;
            font-weight: 800 !important;
        }

        .agent-thought { background-color: #ECFDF5; border-left: 4px solid #059669; padding: 12px 16px; margin-bottom: 10px; border-radius: 6px; color: #065F46; }
        .agent-action { background-color: #F0F9FF; border-left: 4px solid #0284C7; padding: 12px 16px; margin-bottom: 10px; border-radius: 6px; color: #0369A1; }
        .chat-bubble-user { background-color: #F1F5F9; border: 1px solid #CBD5E1; padding: 12px 16px; border-radius: 10px; margin-bottom: 10px; color: #0F172A; }
        .chat-bubble-agent { background-color: #ECFDF5; border: 1px solid #A7F3D0; padding: 12px 16px; border-radius: 10px; margin-bottom: 10px; color: #065F46; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet(
        in_channels=3,
        out_channels=1,
        num_classes=len(HAM10000_CLASSES),
        features=[32, 64, 128, 256]
    ).to(device)
    ckpt_path = os.path.join(BASE_DIR, "best_attention_unet.pth")
    ckpt_info = {"loaded": False, "epoch": 0, "best_metrics": {}}
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        sd = state.get("model_state_dict", state)
        model.load_state_dict(sd, strict=False)
        ckpt_info["loaded"] = True
        ckpt_info["epoch"] = state.get("epoch", 0)
        ckpt_info["best_metrics"] = state.get("best_metrics", {})
    model.eval()
    agent = ClinicalCoPilotAgent(model)
    return model, agent, device, ckpt_info


model, agent, device, ckpt_info = load_models()

# Load local HAM10000 manifest
manifest_df = build_local_manifest(get_default_data_dir())

if ckpt_info["loaded"]:
    bm = ckpt_info.get("best_metrics", {})
    dice_val = bm.get("val_dice", 0.0)
    acc_val = bm.get("val_cls_acc", 0.0)
    st.sidebar.success(
        f"✅ **HAM10000 Model Loaded**\n"
        f"• Val Dice: `{dice_val:.3f}` | 7-Class Acc: `{acc_val*100:.1f}%`\n"
        f"• Local HAM10000 Scans: `{len(manifest_df)}`"
    )
else:
    st.sidebar.warning("⚠️ Untrained weights active. Run HAM10000 training in Tab 5 or via `python train.py`.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "agent", "text": "Hello Doctor! I have analyzed the patient's scan using the HAM10000 Multi-Task Attention U-Net. Ask me about the 7-class differential diagnosis, ABCDE criteria, boundary confidence, or what-if parameter shifts."}
    ]

# Sidebar Input Selection
source_options = ["HAM10000 Clinical Dataset Explorer", "Upload Custom Image", "Synthetic Patient Generator"]
if len(manifest_df) == 0:
    source_options = ["Synthetic Patient Generator", "Upload Custom Image", "HAM10000 Clinical Dataset Explorer"]

image_source = st.sidebar.radio("Input Scan Source (Tab 1):", source_options)

gt_mask = None
selected_meta = None
patient_case_id = "PAT-2026-8891"

if image_source == "HAM10000 Clinical Dataset Explorer":
    if len(manifest_df) == 0:
        st.sidebar.info("No local HAM10000 images found yet. Click below or open Tab 5 to download & train.")
        if st.sidebar.button("⬇️ Download HAM10000 Subset Now"):
            with st.spinner("Downloading real HAM10000 scans & Tschandl masks..."):
                download_ham10000_dataset(samples_per_class=25)
                st.rerun()
        generator = SyntheticDermoscopyGenerator(img_size=(256, 256))
        img_np, gt_mask = generator.generate_sample(seed=42)
    else:
        class_filter_options = ["All 7 Classes"] + [
            f"{c.upper()} — {HAM10000_CLASS_INFO[c]['name']}" for c in HAM10000_CLASSES
        ]
        chosen_cls_filter = st.sidebar.selectbox("Filter by Ground-Truth HAM10000 Class:", class_filter_options)
        if chosen_cls_filter == "All 7 Classes":
            filtered_df = manifest_df
        else:
            cls_code = chosen_cls_filter.split(" — ")[0].lower()
            filtered_df = manifest_df[manifest_df["dx"] == cls_code]
            if len(filtered_df) == 0:
                filtered_df = manifest_df

        sample_labels = [
            f"{row['image_id']} ({row['dx'].upper()} | {row.get('sex', 'u')}/{int(row['age']) if pd.notna(row.get('age')) else '?'}y | {row.get('localization', 'skin')})"
            for _, row in filtered_df.iterrows()
        ]
        chosen_idx = st.sidebar.selectbox("Select HAM10000 Patient Case:", range(len(sample_labels)), format_func=lambda i: sample_labels[i])
        sel_row = filtered_df.iloc[chosen_idx]
        patient_case_id = str(sel_row["image_id"])

        pil_img = Image.open(sel_row["image_path"]).convert("RGB").resize((256, 256), Image.BILINEAR)
        img_np = np.array(pil_img, dtype=np.uint8)

        if isinstance(sel_row.get("mask_path"), str) and os.path.exists(sel_row["mask_path"]):
            pil_m = Image.open(sel_row["mask_path"]).convert("L").resize((256, 256), Image.NEAREST)
            gt_mask = (np.array(pil_m) > 127).astype(np.float32)

        selected_meta = {
            "image_id": str(sel_row["image_id"]),
            "lesion_id": str(sel_row["lesion_id"]),
            "dx": str(sel_row["dx"]),
            "dx_name": HAM10000_CLASS_INFO[str(sel_row["dx"])]["name"],
            "dx_type": str(sel_row.get("dx_type", "histo")),
            "age": f"{int(sel_row['age'])}" if pd.notna(sel_row.get("age")) else "N/A",
            "sex": str(sel_row.get("sex", "unknown")).capitalize(),
            "localization": str(sel_row.get("localization", "unknown")).title()
        }

elif image_source == "Synthetic Patient Generator":
    sample_seed = st.sidebar.slider("Patient Case ID / Seed", 0, 100, 42)
    patient_case_id = f"SYN-2026-{sample_seed:04d}"
    generator = SyntheticDermoscopyGenerator(img_size=(256, 256))
    img_np, gt_mask = generator.generate_sample(seed=sample_seed)
else:
    uploaded_file = st.sidebar.file_uploader("Upload Dermoscopy Image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="main_upload")
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(pil_img.resize((256, 256)))
        patient_case_id = f"UPL-{os.path.splitext(uploaded_file.name)[0]}"
    else:
        generator = SyntheticDermoscopyGenerator(img_size=(256, 256))
        img_np, gt_mask = generator.generate_sample(seed=42)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Research References")
st.sidebar.caption(
    "• **HAM10000 Dataset**: Tschandl et al., *Scientific Data* 2018\n"
    "• **Attention U-Net**: Oktay et al., *MIDL* 2018\n"
    "• **U-Net**: Ronneberger et al., *MICCAI* 2015\n"
    "• **ABCDE Clinical Rule**: Stolz et al."
)

# Header
st.markdown('<div class="main-header">🩺 MedVision-Agent (HAM10000 Edition)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Task Attention U-Net Segmentation, HAM10000 7-Class Diagnostic Classifier & Autonomous Clinical Co-Pilot</div>', unsafe_allow_html=True)

# Run Inference on Primary Scan
img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

analysis = agent.analyze_case(img_np, img_tensor)
pred_mask = analysis["mask"]
att_map = analysis["attention_map"]
var_map = analysis["variance_map"]
cls_res = analysis["classification"]
bio = analysis["biomarkers"]
unc = analysis["uncertainty"]

overlay = img_np.copy()
if gt_mask is not None:
    gt_contours, _ = cv2.findContours((gt_mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, gt_contours, -1, (34, 197, 94), 2)
contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

# Compute Ground-Truth Dice score if GT mask is available
sample_dice = None
if gt_mask is not None:
    p_bin = (pred_mask > 0.5).astype(np.float32)
    g_bin = (gt_mask > 0.5).astype(np.float32)
    inter = float(np.sum(p_bin * g_bin))
    sample_dice = (2.0 * inter + 1e-6) / (float(np.sum(p_bin) + np.sum(g_bin)) + 1e-6)

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 1. HAM10000 Diagnosis & ABCDE Scorecard",
    "🔄 2. Lesion Evolution Tracking (E in ABCDE)",
    "💬 3. Interactive Doctor AI Chatbot",
    "🤖 4. Autonomous Agent ReAct Trace",
    "🏋️ 5. HAM10000 Training & Benchmarks"
])

with tab1:
    if selected_meta is not None:
        st.info(
            f"📋 **HAM10000 Clinical Case `{selected_meta['image_id']}`** (Lesion `{selected_meta['lesion_id']}`) — "
            f"**Ground-Truth Pathology:** `{selected_meta['dx_name']} ({selected_meta['dx'].upper()})` | "
            f"**Verification:** `{selected_meta['dx_type']}` | **Patient:** `{selected_meta['sex']}, Age {selected_meta['age']}` | "
            f"**Anatomical Site:** `{selected_meta['localization']}`"
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("##### (A) Original Dermoscopy")
        st.image(img_np, use_container_width=True)
        st.caption(f"Case ID: {patient_case_id}")
    with col2:
        st.markdown("##### (B) Attention Gate Map")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(att_map, cmap="inferno")
        ax.axis("off")
        fig.patch.set_alpha(0.0)
        st.pyplot(fig, use_container_width=True)
        st.caption("Attention Weights (Suppresses hair/vignette)")
    with col3:
        st.markdown("##### (C) Predicted Mask")
        st.image((pred_mask * 255).astype(np.uint8), use_container_width=True)
        if sample_dice is not None:
            st.caption(f"Coverage: {bio['area_percentage']}% | GT Dice: {sample_dice*100:.1f}%")
        else:
            st.caption(f"Lesion Coverage: {bio['area_percentage']}%")
    with col4:
        st.markdown("##### (D) Boundary Overlay")
        st.image(overlay, use_container_width=True)
        if gt_mask is not None:
            st.caption("🔴 Red = Predicted | 🟢 Green = HAM10000 GT")
        else:
            st.caption("🔴 Red = Predicted Contour Localization")

    st.markdown("---")
    st.markdown("### 🧬 HAM10000 7-Class Diagnostic Classification")
    hcol1, hcol2 = st.columns([1.15, 1.85])
    with hcol1:
        st.metric(
            "HAM10000 Primary Prediction",
            f"{cls_res['predicted_code'].upper()} ({cls_res['confidence_pct']}%)",
            delta=cls_res["predicted_name"]
        )
        st.markdown(f"**Pathology Tier:** `{cls_res['malignancy']}`")
        st.markdown(f"**Clinical Action:** {cls_res['clinical_action']}")
        st.markdown("**Top-3 Differential Breakdown:**")
        for item in cls_res["top3_differential"]:
            st.markdown(f"- **{item['code'].upper()}** (*{item['name']}*): `{item['prob_pct']}%`")

    with hcol2:
        probs_df = pd.DataFrame([
            {
                "Code": code.upper(),
                "Diagnosis": HAM10000_CLASS_INFO[code]["name"],
                "Probability (%)": cls_res["probabilities"][code]
            }
            for code in HAM10000_CLASSES
        ]).sort_values("Probability (%)", ascending=True)

        fig_cls, ax_cls = plt.subplots(figsize=(6.5, 2.8))
        colors = ["#0EA5E9" if c != cls_res["predicted_code"].upper() else "#10B981" for c in probs_df["Code"]]
        bars = ax_cls.barh(probs_df["Code"] + " (" + probs_df["Diagnosis"].str.slice(0, 22) + ")", probs_df["Probability (%)"], color=colors)
        ax_cls.set_xlim(0, 105)
        ax_cls.set_xlabel("Softmax Confidence (%)")
        ax_cls.grid(axis="x", linestyle="--", alpha=0.35)
        for bar in bars:
            w = bar.get_width()
            ax_cls.text(w + 1.2, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_cls, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Clinical Biomarkers (Stolz ABCDE Scorecard)")

    mcol1, mcol2, mcol3, mcol4, mcol5, mcol6 = st.columns(6)
    mcol1.metric("Asymmetry (A)", f"{bio['asymmetry_index']} / 2.0")
    mcol2.metric("Border (B)", f"{bio['border_irregularity']} / 8.0")
    mcol3.metric("Color (C)", f"{bio['color_score']} / 6.0")
    mcol4.metric("Diameter (D)", f"{bio['diameter_mm']} mm")
    mcol5.metric("Evolution (E)", "Active Tracking", delta="Baseline Scan")
    mcol6.metric("TDS Score", f"{bio['tds_score']}", delta=bio['risk_level'])

    st.markdown("---")
    st.markdown(analysis["clinical_report"])

    st.markdown("#### 📥 Export Diagnostic Reports:")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        pdf_data = generate_pdf_report(img_np, pred_mask, overlay, bio, unc, cls_res=cls_res, patient_id=patient_case_id)
        st.download_button(
            label="📄 Download Hospital-Grade PDF Report",
            data=pdf_data,
            file_name=f"MedVision_HAM10000_Report_{patient_case_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with dcol2:
        st.download_button(
            label="📥 Download Markdown Summary",
            data=analysis["clinical_report"],
            file_name=f"medvision_ham10000_report_{patient_case_id}.md",
            mime="text/markdown",
            use_container_width=True
        )

with tab2:
    st.markdown("### 🔄 Longitudinal Lesion Tracking (The 'E' in ABCDE: Evolution)")
    st.markdown("Track and map morphological changes between two timepoints (**Month 0 Baseline** vs. **Month 6 Follow-Up**):")

    evo_mode = st.radio(
        "Select Evolution Tracking Method:",
        ["📁 Upload 2 Patient Photos (Month 0 & Month 6)", "🎲 Synthetic Growth Simulator"],
        horizontal=True
    )

    if evo_mode == "📁 Upload 2 Patient Photos (Month 0 & Month 6)":
        ucol1, ucol2 = st.columns(2)
        with ucol1:
            st.markdown("##### 📤 Step 1: Upload Baseline Scan (Month 0 / T0)")
            file_t0 = st.file_uploader("Upload Month 0 Photo", type=["png", "jpg", "jpeg"], key="upload_t0")
            if file_t0 is not None:
                pil_t0 = Image.open(file_t0).convert("RGB")
                img_t0 = np.array(pil_t0.resize((256, 256)))
            else:
                img_t0 = img_np.copy()

        with ucol2:
            st.markdown("##### 📤 Step 2: Upload Follow-Up Scan (Month 6 / T1)")
            file_t1 = st.file_uploader("Upload Month 6 Photo", type=["png", "jpg", "jpeg"], key="upload_t1")
            if file_t1 is not None:
                pil_t1 = Image.open(file_t1).convert("RGB")
                img_t1 = np.array(pil_t1.resize((256, 256)))
            else:
                gen_def1 = SyntheticDermoscopyGenerator(img_size=(256, 256))
                img_t1, _ = gen_def1.generate_sample(seed=42, scale_factor=1.25, irregularity_factor=1.2)

        img_t0_tensor = torch.from_numpy(img_t0).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        img_t1_tensor = torch.from_numpy(img_t1).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        analysis_t0 = agent.analyze_case(img_t0, img_t0_tensor)
        analysis_t1 = agent.analyze_case(img_t1, img_t1_tensor)

        mask_t0 = analysis_t0["mask"]
        bio_t0 = analysis_t0["biomarkers"]
        mask_t1 = analysis_t1["mask"]
        bio_t1 = analysis_t1["biomarkers"]

        long_res = compare_longitudinal_scans(img_t0, mask_t0, img_t1, mask_t1, bio_t0, bio_t1)

        overlay_t0 = img_t0.copy()
        cnts0, _ = cv2.findContours(mask_t0.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_t0, cnts0, -1, (255, 0, 0), 2)

        overlay_t1 = img_t1.copy()
        cnts1, _ = cv2.findContours(mask_t1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_t1, cnts1, -1, (255, 0, 0), 2)

        st.markdown(f"#### 🧬 Evolution Status: `{long_res['evo_status']}`")
        st.info(f"**Clinical Assessment:** {long_res['evo_reco']}")

        lcol1, lcol2, lcol3 = st.columns(3)
        with lcol1:
            st.markdown("##### 📅 Month 0 Baseline Scan")
            st.image(overlay_t0, use_container_width=True)
            st.caption(f"Area: {bio_t0['area_percentage']}% | Diam: {bio_t0['diameter_mm']}mm | TDS: {bio_t0['tds_score']}")
        with lcol2:
            st.markdown("##### 📅 Month 6 Follow-Up Scan")
            st.image(overlay_t1, use_container_width=True)
            st.caption(f"Area: {bio_t1['area_percentage']}% | Diam: {bio_t1['diameter_mm']}mm | TDS: {bio_t1['tds_score']}")
        with lcol3:
            st.markdown("##### 🗺️ Evolution Growth Map (Mapped)")
            st.image(long_res["growth_vis"], use_container_width=True)
            st.caption("🟢 Green = Stable Core | 🔴 Red = Newly Expanded Margins | 🔵 Blue = Regression")

        st.markdown("---")
        ecol1, ecol2, ecol3, ecol4 = st.columns(4)
        ecol1.metric("Area Change", f"{long_res['area_delta_pct']:+}%", delta=f"{long_res['growth_rate_pct']:+}% Growth Rate")
        ecol2.metric("Diameter Change", f"{long_res['diam_delta_mm']:+} mm")
        ecol3.metric("TDS Score Shift", f"{long_res['tds_delta']:+}")
        ecol4.metric("New Expansion Pixels", f"{long_res['new_pixels']} px")

    else:
        prog_col1, prog_col2 = st.columns([1, 2])
        with prog_col1:
            growth_slider = st.slider("Simulate Lesion Progression over 6 Months (% Growth)", -30, 60, 25, 5)
            scale_mod = 1.0 + (growth_slider / 100.0)
            irreg_mod = 1.0 + (max(0, growth_slider) / 100.0 * 0.5)

            gen_followup = SyntheticDermoscopyGenerator(img_size=(256, 256))
            img_t1, _ = gen_followup.generate_sample(seed=42, scale_factor=scale_mod, irregularity_factor=irreg_mod)
            img_t1_tensor = torch.from_numpy(img_t1).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

            analysis_t1 = agent.analyze_case(img_t1, img_t1_tensor)
            mask_t1 = analysis_t1["mask"]
            bio_t1 = analysis_t1["biomarkers"]

            long_res = compare_longitudinal_scans(img_np, pred_mask, img_t1, mask_t1, bio, bio_t1)

        with prog_col2:
            st.markdown(f"#### 🧬 Evolution Status: `{long_res['evo_status']}`")
            st.info(f"**Clinical Assessment:** {long_res['evo_reco']}")

        lcol1, lcol2, lcol3 = st.columns(3)
        with lcol1:
            st.markdown("##### 📅 Baseline (Month 0)")
            st.image(img_np, use_container_width=True)
            st.caption(f"Area: {bio['area_percentage']}% | Diam: {bio['diameter_mm']}mm | TDS: {bio['tds_score']}")
        with lcol2:
            st.markdown("##### 📅 Follow-Up (Month 6)")
            st.image(img_t1, use_container_width=True)
            st.caption(f"Area: {bio_t1['area_percentage']}% | Diam: {bio_t1['diameter_mm']}mm | TDS: {bio_t1['tds_score']}")
        with lcol3:
            st.markdown("##### 🗺️ Evolution Growth Map")
            st.image(long_res["growth_vis"], use_container_width=True)
            st.caption("🟢 Green = Stable Core | 🔴 Red = New Expansion | 🔵 Blue = Regression")

        st.markdown("---")
        ecol1, ecol2, ecol3, ecol4 = st.columns(4)
        ecol1.metric("Area Change", f"{long_res['area_delta_pct']:+}%", delta=f"{long_res['growth_rate_pct']:+}% Growth Rate")
        ecol2.metric("Diameter Change", f"{long_res['diam_delta_mm']:+} mm")
        ecol3.metric("TDS Score Shift", f"{long_res['tds_delta']:+}")
        ecol4.metric("New Expansion Pixels", f"{long_res['new_pixels']} px")

with tab3:
    st.markdown("### 💬 Interactive Doctor-in-the-Loop Clinical AI Chatbot")
    st.markdown("Ask the Clinical Co-Pilot about HAM10000 7-class differential probabilities, test 'what-if' diameter scenarios, or query surgical protocols:")

    st.markdown("##### ⚡ Quick Prompt Buttons (Click to ask):")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)

    chat_agent = ClinicalChatAgent(analysis)

    prompt_to_send = None
    if qcol1.button("🧬 HAM10000 7-Class Differential"):
        prompt_to_send = "Explain the HAM10000 7-class differential diagnosis for this scan."
    if qcol2.button("❓ Why " + bio['risk_level'].split()[0] + " Risk?"):
        prompt_to_send = "Why was this scan classified as " + bio['risk_level'] + "?"
    if qcol3.button("🔬 What if diameter = 4.0 mm?"):
        prompt_to_send = "What if the diameter was 4.0 mm instead?"
    if qcol4.button("📋 Surgical & Biopsy Protocol"):
        prompt_to_send = "What surgical protocol or biopsy margin is recommended?"

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user"><b>👨‍⚕️ Doctor:</b> {msg["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-agent"><b>🤖 Clinical Co-Pilot:</b> {msg["text"]}</div>', unsafe_allow_html=True)

    user_input = st.chat_input("Type your question for the AI Clinical Co-Pilot here...")

    active_prompt = prompt_to_send if prompt_to_send else user_input
    if active_prompt:
        st.session_state.chat_history.append({"role": "user", "text": active_prompt})
        response = chat_agent.reply(active_prompt)
        st.session_state.chat_history.append({"role": "agent", "text": response})
        st.rerun()

with tab4:
    st.markdown("### 🤖 Autonomous Agentic Workflow & ReAct Trace")
    st.markdown("The Clinical AI Agent reasons step-by-step across 5 specialized tools (including HAM10000 Segmentation & 7-Class Taxonomy) and synthesizes findings:")

    for log in analysis["reasoning_logs"]:
        if log["type"] == "thought":
            st.markdown(f"""
            <div class="agent-thought">
                <b>💭 Agent Thought (Step {log['step']}):</b><br>
                {log['message']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="agent-action">
                <b>⚙️ Tool Call: <code>{log['tool']}</code> (Step {log['step']}):</b><br>
                {log['message']}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔍 Monte Carlo Dropout Boundary Uncertainty Map")
    ucol1, ucol2 = st.columns([1, 2])
    with ucol1:
        fig_unc, ax_unc = plt.subplots(figsize=(4, 4))
        im = ax_unc.imshow(var_map, cmap="magma")
        plt.colorbar(im, ax=ax_unc, fraction=0.046, pad=0.04)
        ax_unc.axis("off")
        fig_unc.patch.set_alpha(0.0)
        st.pyplot(fig_unc, use_container_width=True)
    with ucol2:
        st.markdown(f"""
        * **Boundary Confidence Score**: `{unc['confidence_score']}%`
        * **Edge Ambiguity Metric**: `{unc['edge_uncertainty']}`
        * **Clinical Interpretation**: Warmer pixel colors indicate regions where the neural network exhibits high epistemic variance (usually at ambiguous lesion borders).
        """)

with tab5:
    st.markdown("### 🏋️ HAM10000 Dataset Training, Evaluation & Ablation Dashboard")
    st.markdown(
        "Train or fine-tune the **Multi-Task Attention U-Net** (Joint Tschandl Lesion Segmentation + 7-Class HAM10000 Classification) "
        "and view real validation metrics, training curves, and Baseline U-Net vs. Attention U-Net ablation benchmarks."
    )

    with st.expander("⚙️ Configure & Launch Interactive HAM10000 Training Run", expanded=not ckpt_info["loaded"]):
        tcol1, tcol2, tcol3, tcol4 = st.columns(4)
        ui_epochs = tcol1.number_input("Training Epochs", min_value=1, max_value=100, value=10, step=1)
        ui_batch = tcol2.selectbox("Batch Size", [4, 8, 16, 32], index=1)
        ui_lr = tcol3.select_slider("Learning Rate", options=[1e-4, 3e-4, 5e-4, 1e-3, 2e-3], value=1e-3)
        ui_spc = tcol4.number_input("Samples / Class (7 classes)", min_value=5, max_value=1500, value=25, step=5)

        if st.button("🚀 Start HAM10000 Multi-Task Training Now", use_container_width=True):
            prog_bar = st.progress(0.0)
            status_box = st.empty()

            def _ui_cb(msg, pct):
                prog_bar.progress(min(1.0, max(0.0, float(pct))))
                status_box.info(f"⏳ {msg}")

            with st.spinner("Running HAM10000 Multi-Task Attention U-Net training pipeline..."):
                run_ham10000_training(
                    epochs=int(ui_epochs),
                    batch_size=int(ui_batch),
                    lr=float(ui_lr),
                    samples_per_class=int(ui_spc),
                    run_ablation=True,
                    progress_callback=_ui_cb
                )
            st.cache_resource.clear()
            st.success("✅ HAM10000 Training & Ablation Complete! Reloading model weights...")
            st.rerun()

    history_path = os.path.join(BASE_DIR, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            hist = json.load(f)

        bm = hist.get("best_metrics", {})
        st.markdown("#### 🏆 HAM10000 Validation Benchmark Results")
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric("Val Dice (DSC)", f"{bm.get('val_dice', 0)*100:.2f}%")
        b2.metric("Val IoU (Jaccard)", f"{bm.get('val_iou', 0)*100:.2f}%")
        b3.metric("Boundary Precision", f"{bm.get('val_precision', 0)*100:.2f}%")
        b4.metric("Boundary Recall", f"{bm.get('val_recall', 0)*100:.2f}%")
        b5.metric("7-Class Accuracy", f"{bm.get('val_cls_acc', 0)*100:.2f}%")
        b6.metric("Macro F1-Score", f"{bm.get('val_cls_f1', 0)*100:.2f}%")

        epochs_list = hist.get("epochs", [])
        if len(epochs_list) > 0:
            df_ep = pd.DataFrame(epochs_list)
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown("##### 📈 Multi-Task Loss & Segmentation Dice Curve")
                fig1, ax1 = plt.subplots(figsize=(5.5, 3.2))
                ax1.plot(df_ep["epoch"], df_ep["train_loss"], label="Train Total Loss", linestyle="--", color="#64748B")
                ax1.plot(df_ep["epoch"], df_ep["val_loss"], label="Val Total Loss", color="#EF4444", linewidth=2)
                ax1.plot(df_ep["epoch"], df_ep["val_dice"], label="Val Dice (DSC)", color="#10B981", linewidth=2.2)
                ax1.plot(df_ep["epoch"], df_ep["val_iou"], label="Val IoU (Jaccard)", color="#0EA5E9", linewidth=1.8)
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Metric / Loss")
                ax1.grid(True, linestyle="--", alpha=0.35)
                ax1.legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig1, use_container_width=True)

            with ch2:
                st.markdown("##### 🎯 HAM10000 7-Class Accuracy & Macro F1 Curve")
                fig2, ax2 = plt.subplots(figsize=(5.5, 3.2))
                ax2.plot(df_ep["epoch"], df_ep["train_cls_acc"] * 100.0, label="Train 7-Class Acc (%)", linestyle="--", color="#64748B")
                ax2.plot(df_ep["epoch"], df_ep["val_cls_acc"] * 100.0, label="Val 7-Class Acc (%)", color="#0284C7", linewidth=2.2)
                ax2.plot(df_ep["epoch"], df_ep["val_cls_f1"] * 100.0, label="Val Macro F1 (%)", color="#8B5CF6", linewidth=2)
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Percentage (%)")
                ax2.grid(True, linestyle="--", alpha=0.35)
                ax2.legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)

        st.markdown("---")
        ab = hist.get("ablation", {})
        col_ab1, col_ab2 = st.columns(2)
        with col_ab1:
            st.markdown("#### 🔬 Scientific Ablation Study (HAM10000 Validation Split)")
            if "baseline_unet" in ab and "attention_unet" in ab:
                base_m = ab["baseline_unet"]
                att_m = ab["attention_unet"]
                d_dice = (att_m["val_dice"] - base_m["val_dice"]) * 100.0
                d_iou = (att_m["val_iou"] - base_m["val_iou"]) * 100.0
                d_prec = (att_m["val_precision"] - base_m["val_precision"]) * 100.0
                d_acc = (att_m["val_cls_acc"] - base_m["val_cls_acc"]) * 100.0
                st.markdown(f"""
                | Metric | Baseline U-Net | Attention U-Net (HAM10000) | Delta |
                | :--- | :--- | :--- | :--- |
                | **Dice Score (DSC)** | `{base_m['val_dice']:.4f}` | **`{att_m['val_dice']:.4f}`** | 🟢 **`{d_dice:+.2f}%`** |
                | **IoU (Jaccard Index)** | `{base_m['val_iou']:.4f}` | **`{att_m['val_iou']:.4f}`** | 🟢 **`{d_iou:+.2f}%`** |
                | **Precision** | `{base_m['val_precision']:.4f}` | **`{att_m['val_precision']:.4f}`** | 🟢 **`{d_prec:+.2f}%`** |
                | **7-Class Accuracy** | `{base_m['val_cls_acc']*100:.1f}%` | **`{att_m['val_cls_acc']*100:.1f}%`** | 🟢 **`{d_acc:+.1f}%`** |
                | **Parameters** | `{base_m['params_m']}M` | `{att_m['params_m']}M` | Multi-Task Head |
                """)
            else:
                st.markdown("Run training with ablation enabled to populate comparison table.")

        with col_ab2:
            st.markdown("#### 🧬 Per-Class HAM10000 Diagnostic Summary")
            per_cls = bm.get("per_class_acc", {})
            if per_cls:
                cls_rows = []
                for c in HAM10000_CLASSES:
                    cls_rows.append({
                        "Code": c.upper(),
                        "Disease Name": HAM10000_CLASS_INFO[c]["name"],
                        "Malignancy": HAM10000_CLASS_INFO[c]["malignancy"],
                        "Val Acc (%)": f"{per_cls.get(c, 0.0):.1f}%"
                    })
                st.dataframe(pd.DataFrame(cls_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No `training_history.json` found yet. Click **Start HAM10000 Multi-Task Training Now** above or run `python train.py`.")
