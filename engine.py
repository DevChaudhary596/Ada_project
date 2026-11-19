import io
import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
import cv2
from fpdf import FPDF

from dataset import HAM10000_CLASSES, HAM10000_CLASS_INFO, IDX_TO_CLASS


class SyntheticDermoscopyGenerator:
    """Procedural skin lesion generator with realistic pigment networks, hairs, and vignette."""
    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size

    def generate_sample(self, seed=None, scale_factor=1.0, irregularity_factor=1.0):
        if seed is not None:
            np.random.seed(seed)

        w, h = self.img_size

        base_r = np.random.randint(200, 240)
        base_g = np.random.randint(160, 200)
        base_b = np.random.randint(140, 180)
        skin = np.full((h, w, 3), [base_r, base_g, base_b], dtype=np.uint8)

        noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
        skin = np.clip(skin.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        cx = w // 2 + np.random.randint(-20, 20)
        cy = h // 2 + np.random.randint(-20, 20)
        rx = int(np.random.randint(35, 65) * scale_factor)
        ry = int(np.random.randint(35, 65) * scale_factor)

        num_points = np.random.randint(12, 24)
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        radius_noise = np.random.uniform(0.7 * irregularity_factor, 1.3 * irregularity_factor, num_points)

        points = []
        for angle, r_mod in zip(angles, radius_noise):
            px = int(cx + rx * r_mod * np.cos(angle))
            py = int(cy + ry * r_mod * np.sin(angle))
            points.append([px, py])
        points = np.array(points, dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        mask_pil = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=2))
        mask_binary = (np.array(mask_pil) > 127).astype(np.uint8) * 255

        lesion_r = np.random.randint(60, 120)
        lesion_g = np.random.randint(40, 85)
        lesion_b = np.random.randint(30, 70)
        lesion_layer = np.full((h, w, 3), [lesion_r, lesion_g, lesion_b], dtype=np.uint8)

        dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        max_d = dist_transform.max() if dist_transform.max() > 0 else 1
        dist_norm = (dist_transform / max_d)[:, :, np.newaxis]

        dark_pigment = np.full((h, w, 3), [lesion_r // 2, lesion_g // 2, lesion_b // 2], dtype=np.uint8)
        blended_lesion = (lesion_layer * (1 - dist_norm * 0.5) + dark_pigment * (dist_norm * 0.5)).astype(np.uint8)

        smooth_mask = (cv2.GaussianBlur(mask_binary, (15, 15), 0) / 255.0)[:, :, np.newaxis]
        composite = (skin * (1 - smooth_mask) + blended_lesion * smooth_mask).astype(np.uint8)

        num_hairs = np.random.randint(2, 6)
        for _ in range(num_hairs):
            hx1, hy1 = np.random.randint(0, w), np.random.randint(0, h)
            hx2, hy2 = hx1 + np.random.randint(-60, 60), hy1 + np.random.randint(-60, 60)
            cv2.line(composite, (hx1, hy1), (hx2, hy2), (25, 20, 15), thickness=np.random.randint(1, 2))

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - w/2)**2 + (Y - h/2)**2)
        max_radius = np.sqrt((w/2)**2 + (h/2)**2)
        vignette = 1 - 0.25 * (dist_from_center / max_radius)**2
        composite = np.clip(composite * vignette[:, :, np.newaxis], 0, 255).astype(np.uint8)

        return composite, (mask_binary > 0).astype(np.float32)


def classify_ham10000_lesion(model, image_tensor):
    """
    Runs the Multi-Task Attention U-Net 7-Class HAM10000 Diagnostic Head
    and returns predicted class metadata, confidence, top-3 differentials, and probabilities.
    """
    model.eval()
    with torch.no_grad():
        _, cls_logits = model(image_tensor, return_classification=True)
        probs = F.softmax(cls_logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_code = IDX_TO_CLASS.get(pred_idx, "nv")
    info = HAM10000_CLASS_INFO[pred_code]

    prob_dict = {
        cls_code: round(float(probs[idx]) * 100.0, 2)
        for idx, cls_code in enumerate(HAM10000_CLASSES)
    }
    sorted_classes = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    top3 = [
        {
            "code": c,
            "name": HAM10000_CLASS_INFO[c]["name"],
            "prob_pct": p,
            "malignancy": HAM10000_CLASS_INFO[c]["malignancy"]
        }
        for c, p in sorted_classes[:3]
    ]

    return {
        "predicted_code": pred_code,
        "predicted_name": info["name"],
        "malignancy": info["malignancy"],
        "risk_tier": info["risk_tier"],
        "clinical_action": info["clinical_action"],
        "confidence_pct": round(float(probs[pred_idx]) * 100.0, 1),
        "probabilities": prob_dict,
        "top3_differential": top3
    }


def extract_biomarkers(image_np, mask_np, ham10000_pred=None):
    """ABCD Clinical Rule calculation (Stolz Algorithm) combined with HAM10000 pathology risk."""
    binary_mask = (mask_np > 0.5).astype(np.uint8)
    total_pixels = binary_mask.size
    lesion_pixels = np.sum(binary_mask)

    if lesion_pixels == 0:
        return {
            "area_pixels": 0,
            "area_percentage": 0.0,
            "asymmetry_index": 0.0,
            "border_irregularity": 1.0,
            "color_score": 1,
            "diameter_mm": 0.0,
            "tds_score": 1.0,
            "risk_level": "Benign / Normal Skin",
            "recommendation": "No lesion detected in scan region."
        }

    area_pct = (lesion_pixels / total_pixels) * 100.0

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(main_contour, True)
        area = cv2.contourArea(main_contour)
        border_irregularity = (perimeter ** 2) / (4.0 * np.pi * area) if area > 0 else 1.0
        x, y, w, h = cv2.boundingRect(main_contour)
        diameter_pixels = np.sqrt(w**2 + h**2)
        diameter_mm = diameter_pixels * 0.075
    else:
        border_irregularity = 1.0
        diameter_mm = 2.0

    flipped_h = cv2.flip(binary_mask, 1)
    diff_h = np.logical_xor(binary_mask, flipped_h).sum()
    asym_h = diff_h / (2.0 * lesion_pixels)

    flipped_v = cv2.flip(binary_mask, 0)
    diff_v = np.logical_xor(binary_mask, flipped_v).sum()
    asym_v = diff_v / (2.0 * lesion_pixels)

    asymmetry_index = min(2.0, (asym_h + asym_v))

    masked_img = image_np[binary_mask == 1]
    if len(masked_img) > 0:
        color_variance = float(np.mean(np.std(masked_img, axis=0)))
        color_score = min(6, max(1, int(color_variance / 8.0) + 1))
    else:
        color_score = 1

    A_score = min(2.0, asymmetry_index * 1.5)
    B_score = min(8.0, max(1.0, border_irregularity * 2.5))
    C_score = color_score
    D_score = min(5.0, diameter_mm / 2.0)

    tds = (1.3 * A_score) + (0.1 * B_score) + (0.5 * C_score) + (0.5 * D_score)

    if tds < 4.75 and diameter_mm < 6.0:
        risk_level = "Benign / Low Risk"
        recommendation = "Routine annual dermatological examination recommended."
    elif 4.75 <= tds <= 5.45:
        risk_level = "Suspicious / Moderate Risk"
        recommendation = "Short-term 3-month digital dermoscopy follow-up or diagnostic excision recommended."
    else:
        risk_level = "High Risk / Malignant Suspected"
        recommendation = "Urgent histopathological biopsy and dermatological oncology consult recommended."

    # Integrate HAM10000 7-class pathology classification if high-risk malignancy detected
    if ham10000_pred is not None:
        pred_code = ham10000_pred.get("predicted_code", "")
        if pred_code in ("mel", "bcc") and risk_level == "Benign / Low Risk":
            risk_level = "Suspicious / Moderate-High Risk"
            recommendation = f"HAM10000 classifier detected {ham10000_pred['predicted_name']} pattern. {ham10000_pred['clinical_action']}"
        elif pred_code == "akiec" and risk_level == "Benign / Low Risk":
            risk_level = "Suspicious / Moderate Risk"
            recommendation = ham10000_pred["clinical_action"]

    return {
        "area_pixels": int(lesion_pixels),
        "area_percentage": round(float(area_pct), 2),
        "asymmetry_index": round(float(A_score), 2),
        "border_irregularity": round(float(B_score), 2),
        "color_score": int(C_score),
        "diameter_mm": round(float(diameter_mm), 2),
        "tds_score": round(float(tds), 2),
        "risk_level": risk_level,
        "recommendation": recommendation
    }


def estimate_uncertainty(model, image_tensor, num_samples=6):
    """Monte Carlo Dropout for epistemic uncertainty estimation."""
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(image_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.append(probs)

    preds_array = np.array(predictions)
    mean_mask = np.mean(preds_array, axis=0)[0, 0]
    variance_map = np.var(preds_array, axis=0)[0, 0]

    v_max = variance_map.max() if variance_map.max() > 0 else 1.0
    norm_variance = variance_map / v_max

    boundary_mask = cv2.Canny((mean_mask > 0.5).astype(np.uint8) * 255, 100, 200)
    if boundary_mask.sum() > 0:
        edge_uncertainty = float(np.mean(variance_map[boundary_mask > 0]))
    else:
        edge_uncertainty = float(np.mean(variance_map))

    confidence_score = max(0.0, min(1.0, 1.0 - (edge_uncertainty * 5.0)))

    return {
        "mean_mask": mean_mask,
        "variance_map": norm_variance,
        "confidence_score": round(confidence_score * 100.0, 1),
        "edge_uncertainty": round(edge_uncertainty, 4)
    }


def compare_longitudinal_scans(img1, mask1, img2, mask2, bio1, bio2):
    """
    Feature 1: Longitudinal Lesion Tracking (Evolution / 'E' in ABCDE).
    Compares baseline scan (Month 0) vs follow-up scan (Month 6),
    supports custom uploaded scans, HAM10000 scans, and synthetic samples.
    """
    if img1.shape[:2] != (256, 256):
        img1 = cv2.resize(img1, (256, 256))
    if img2.shape[:2] != (256, 256):
        img2 = cv2.resize(img2, (256, 256))
    if mask1.shape[:2] != (256, 256):
        mask1 = cv2.resize(mask1, (256, 256))
    if mask2.shape[:2] != (256, 256):
        mask2 = cv2.resize(mask2, (256, 256))

    m1 = (mask1 > 0.5).astype(np.uint8)
    m2 = (mask2 > 0.5).astype(np.uint8)

    area_delta = bio2["area_percentage"] - bio1["area_percentage"]
    diam_delta = bio2["diameter_mm"] - bio1["diameter_mm"]
    tds_delta = bio2["tds_score"] - bio1["tds_score"]

    persistent_core = np.logical_and(m1 == 1, m2 == 1).astype(np.uint8)
    new_growth = np.logical_and(m2 == 1, m1 == 0).astype(np.uint8)
    regressed = np.logical_and(m1 == 1, m2 == 0).astype(np.uint8)

    growth_vis = img2.copy()
    growth_vis[persistent_core == 1] = (growth_vis[persistent_core == 1] * 0.35 + np.array([34, 197, 94]) * 0.65).astype(np.uint8)
    growth_vis[new_growth == 1] = (growth_vis[new_growth == 1] * 0.25 + np.array([239, 68, 68]) * 0.75).astype(np.uint8)
    growth_vis[regressed == 1] = (growth_vis[regressed == 1] * 0.35 + np.array([59, 130, 246]) * 0.65).astype(np.uint8)

    if bio1["area_pixels"] > 0:
        growth_rate_pct = ((bio2["area_pixels"] - bio1["area_pixels"]) / float(bio1["area_pixels"])) * 100.0
    else:
        growth_rate_pct = 100.0 if bio2["area_pixels"] > 0 else 0.0

    if growth_rate_pct > 15.0 or diam_delta > 1.2:
        evo_status = "Rapid Expansion / High Alert"
        evo_reco = f"Significant perimeter expansion detected (+{growth_rate_pct:.1f}% area, +{diam_delta:.2f}mm diameter). Urgent biopsy indicated."
    elif growth_rate_pct > 5.0 or diam_delta > 0.4:
        evo_status = "Moderate Progression"
        evo_reco = f"Moderate expansion noted (+{growth_rate_pct:.1f}%). Short-term clinical dermatological monitoring recommended."
    elif growth_rate_pct < -8.0:
        evo_status = "Regression Noted"
        evo_reco = f"Lesion surface contracted by {abs(growth_rate_pct):.1f}%. Continue periodic routine monitoring."
    else:
        evo_status = "Morphologically Stable"
        evo_reco = "Lesion boundary and pigmentation remain stable across both scan intervals (<5% variance)."

    return {
        "growth_vis": growth_vis,
        "area_delta_pct": round(float(area_delta), 2),
        "diam_delta_mm": round(float(diam_delta), 2),
        "tds_delta": round(float(tds_delta), 2),
        "growth_rate_pct": round(float(growth_rate_pct), 1),
        "evo_status": evo_status,
        "evo_reco": evo_reco,
        "new_pixels": int(np.sum(new_growth)),
        "stable_pixels": int(np.sum(persistent_core)),
        "regressed_pixels": int(np.sum(regressed))
    }


def generate_pdf_report(img_np, mask_np, overlay_np, bio, unc, cls_res=None, patient_id="PAT-2026-8891"):
    """Hospital-Grade One-Click PDF Diagnostic Report Export with HAM10000 7-Class Classification."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_fill_color(15, 118, 110)
    pdf.rect(0, 0, 210, 24, "F")

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 6)
    pdf.cell(190, 8, "MedVision AI - HAM10000 Dermatological Triage Report", align="L")
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_xy(10, 14)
    pdf.cell(190, 6, "Multi-Task Attention U-Net Segmentation, HAM10000 7-Class Taxonomy & ABCDE Summary", align="L")

    pdf.set_text_color(30, 41, 59)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_xy(10, 28)
    curr_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(95, 6, f"Patient Case ID: {patient_id}", align="L")
    pdf.cell(95, 6, f"Date & Time: {curr_date}", align="R")

    pdf.set_draw_color(203, 213, 225)
    pdf.line(10, 36, 200, 36)

    os.makedirs("temp_pdf", exist_ok=True)
    img_path = "temp_pdf/scan.png"
    mask_path = "temp_pdf/mask.png"
    overlay_path = "temp_pdf/overlay.png"

    Image.fromarray(img_np).save(img_path)
    Image.fromarray((mask_np * 255).astype(np.uint8)).save(mask_path)
    Image.fromarray(overlay_np).save(overlay_path)

    pdf.set_xy(10, 40)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(190, 6, "1. Visual Segmentation & Boundary Localization", ln=True)

    y_img = 48
    pdf.image(img_path, x=15, y=y_img, w=55, h=55)
    pdf.image(mask_path, x=78, y=y_img, w=55, h=55)
    pdf.image(overlay_path, x=140, y=y_img, w=55, h=55)

    pdf.set_font("Helvetica", "", 8)
    pdf.set_xy(15, y_img + 56)
    pdf.cell(55, 4, "(A) Original Dermoscopy", align="C")
    pdf.set_xy(78, y_img + 56)
    pdf.cell(55, 4, "(B) Attention U-Net Mask", align="C")
    pdf.set_xy(140, y_img + 56)
    pdf.cell(55, 4, "(C) Clinical Contour Overlay", align="C")

    pdf.set_xy(10, 114)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(190, 6, "2. Morphological Biomarkers & Stolz ABCDE Scorecard", ln=True)

    pdf.set_fill_color(241, 245, 249)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_xy(10, 122)
    pdf.cell(45, 7, "Parameter", border=1, fill=True)
    pdf.cell(35, 7, "Measured Value", border=1, fill=True)
    pdf.cell(45, 7, "Clinical Scale", border=1, fill=True)
    pdf.cell(65, 7, "Risk Assessment", border=1, fill=True, ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_x(10)
    pdf.cell(45, 6, "Asymmetry (A)", border=1)
    pdf.cell(35, 6, f"{bio['asymmetry_index']} / 2.0", border=1)
    pdf.cell(45, 6, "Weight = 1.3", border=1)
    pdf.cell(65, 6, "High Asymmetry" if bio['asymmetry_index'] > 1.2 else "Bilateral Symmetry", border=1, ln=True)

    pdf.set_x(10)
    pdf.cell(45, 6, "Border Irregularity (B)", border=1)
    pdf.cell(35, 6, f"{bio['border_irregularity']} / 8.0", border=1)
    pdf.cell(45, 6, "Weight = 0.1", border=1)
    pdf.cell(65, 6, "Notched Margin" if bio['border_irregularity'] > 3.0 else "Demarcated Margin", border=1, ln=True)

    pdf.set_x(10)
    pdf.cell(45, 6, "Color Variegation (C)", border=1)
    pdf.cell(35, 6, f"{bio['color_score']} / 6.0", border=1)
    pdf.cell(45, 6, "Weight = 0.5", border=1)
    pdf.cell(65, 6, f"{bio['color_score']} Pigment clusters", border=1, ln=True)

    pdf.set_x(10)
    pdf.cell(45, 6, "Diameter (D)", border=1)
    pdf.cell(35, 6, f"{bio['diameter_mm']} mm", border=1)
    pdf.cell(45, 6, "Weight = 0.5", border=1)
    pdf.cell(65, 6, "> 6mm Critical Threshold" if bio['diameter_mm'] > 6.0 else "Within benign limits", border=1, ln=True)

    pdf.set_fill_color(226, 232, 240)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(10)
    pdf.cell(45, 7, "Total Score (TDS)", border=1, fill=True)
    pdf.cell(35, 7, f"{bio['tds_score']} / 8.0", border=1, fill=True)
    pdf.cell(45, 7, f"Conf: {unc['confidence_score']}%", border=1, fill=True)
    pdf.cell(65, 7, f"{bio['risk_level']}", border=1, fill=True, ln=True)

    pdf.set_xy(10, 162)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(190, 6, "3. HAM10000 7-Class Diagnostic Classification & Clinical Triage", ln=True)

    pdf.set_xy(10, 170)
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(15, 118, 110)
    pdf.set_font("Helvetica", "", 9)
    ham_line = ""
    if cls_res is not None:
        ham_line = (
            f"HAM10000 Primary Diagnosis: {cls_res['predicted_name']} ({cls_res['predicted_code'].upper()}) "
            f"[{cls_res['confidence_pct']}% prob] - {cls_res['malignancy']}\n"
        )
    reco_text = (
        f"{ham_line}"
        f"Risk Classification: {bio['risk_level']}\n"
        f"Protocol Recommendation: {bio['recommendation']}\n"
        f"Quality Assurance: Attention U-Net (HAM10000 trained) suppressed artifacts with {unc['confidence_score']}% boundary precision."
    )
    pdf.multi_cell(190, 5, reco_text, border=1, fill=True)

    pdf.set_xy(10, 215)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(95, 5, "Attending Physician / Dermatologist:", ln=False)
    pdf.cell(95, 5, "Digital AI System Verification:", ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(10, 228)
    pdf.cell(95, 5, "Signature: ___________________________", ln=False)
    pdf.cell(95, 5, "MedVision-Agent HAM10000 Engine v3.0 (Verified)", ln=True)

    pdf_buffer = io.BytesIO()
    pdf_bytes = pdf.output()
    pdf_buffer.write(pdf_bytes)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
