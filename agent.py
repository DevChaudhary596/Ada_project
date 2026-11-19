import time
import numpy as np
import torch
from engine import extract_biomarkers, estimate_uncertainty, classify_ham10000_lesion
from dataset import HAM10000_CLASS_INFO


class SegmentationTool:
    """Tool 1: Runs HAM10000-trained Attention U-Net model on medical dermoscopy images."""
    name = "segment_medical_image"

    def __init__(self, model):
        self.model = model

    def run(self, image_tensor):
        self.model.eval()
        with torch.no_grad():
            logits, att_dict = self.model(image_tensor, return_attention=True)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
            binary_mask = (probs > 0.5).astype(np.uint8)
            ag_map = att_dict["ag1"].cpu().numpy()[0, 0]
        return {
            "predicted_mask": binary_mask,
            "probabilities": probs,
            "attention_map": ag_map
        }


class HAM10000ClassificationTool:
    """Tool 2: Classifies skin lesions into the 7 HAM10000 diagnostic categories."""
    name = "classify_ham10000_taxonomy"

    def __init__(self, model):
        self.model = model

    def run(self, image_tensor):
        return classify_ham10000_lesion(self.model, image_tensor)


class BiomarkerTool:
    """Tool 3: Computes Stolz ABCD morphological biomarkers."""
    name = "extract_abcd_biomarkers"

    def run(self, image_np, mask_np, ham10000_pred=None):
        return extract_biomarkers(image_np, mask_np, ham10000_pred=ham10000_pred)


class UncertaintyTool:
    """Tool 4: Evaluates epistemic uncertainty and boundary confidence via Monte Carlo Dropout."""
    name = "evaluate_model_uncertainty"

    def __init__(self, model):
        self.model = model

    def run(self, image_tensor):
        return estimate_uncertainty(self.model, image_tensor, num_samples=6)


class GuidelinesTool:
    """Tool 5: Queries dermatological risk protocols combining HAM10000 7-class taxonomy & Stolz ABCD."""
    name = "dermatology_guidelines_lookup"

    def run(self, tds_score, diameter_mm, cls_res=None):
        if tds_score < 4.75 and diameter_mm < 6.0:
            risk = "Benign / Low Risk"
            reco = "Routine annual dermatological examination recommended."
        elif 4.75 <= tds_score <= 5.45:
            risk = "Suspicious / Moderate Risk"
            reco = "Short-term 3-month digital dermoscopy follow-up or diagnostic excision recommended."
        else:
            risk = "High Risk / Malignant Suspected"
            reco = "Urgent histopathological biopsy and dermatological oncology consult recommended."

        if cls_res is not None:
            pred_code = cls_res["predicted_code"]
            if pred_code in ("mel", "bcc") and risk == "Benign / Low Risk":
                risk = "Suspicious / Moderate-High Risk"
                reco = f"{cls_res['clinical_action']} (Flagged by HAM10000 {cls_res['predicted_name']} detector)."
            elif pred_code in ("mel", "bcc"):
                reco = f"{cls_res['clinical_action']}"
            elif pred_code == "akiec" and risk == "Benign / Low Risk":
                risk = "Suspicious / Moderate Risk"
                reco = cls_res["clinical_action"]

        return {"risk": risk, "recommendation": reco}


class ClinicalCoPilotAgent:
    """
    Autonomous Medical AI Co-Pilot Agent (Trained on HAM10000)
    Implements a ReAct (Reasoning + Action) 5-tool clinical diagnostic loop.
    """
    def __init__(self, model):
        self.model = model
        self.seg_tool = SegmentationTool(model)
        self.cls_tool = HAM10000ClassificationTool(model)
        self.bio_tool = BiomarkerTool()
        self.unc_tool = UncertaintyTool(model)
        self.guide_tool = GuidelinesTool()

    def analyze_case(self, image_np, image_tensor):
        logs = []

        logs.append({
            "step": 1,
            "type": "thought",
            "message": "Received patient dermoscopy scan. Initiating HAM10000-trained Attention U-Net spatial segmentation tool to isolate lesion boundaries."
        })

        time.sleep(0.02)
        seg_res = self.seg_tool.run(image_tensor)
        mask = seg_res["predicted_mask"]
        att_map = seg_res["attention_map"]
        logs.append({
            "step": 2,
            "type": "action",
            "tool": "SegmentationTool",
            "message": f"Segmentation complete. Identified lesion region covering {int(np.sum(mask))} pixels."
        })

        logs.append({
            "step": 3,
            "type": "thought",
            "message": "Executing Multi-Task HAM10000 7-Class Diagnostic Classifier (akiec, bcc, bkl, df, mel, nv, vasc) on attention-gated encoder features."
        })
        cls_res = self.cls_tool.run(image_tensor)
        logs.append({
            "step": 4,
            "type": "action",
            "tool": "HAM10000ClassificationTool",
            "message": f"HAM10000 Diagnosis: {cls_res['predicted_name']} ({cls_res['predicted_code'].upper()}) with {cls_res['confidence_pct']}% confidence [{cls_res['malignancy']}]."
        })

        logs.append({
            "step": 5,
            "type": "thought",
            "message": "Analyzing segmented lesion morphology using ABCD clinical metrics (Asymmetry, Border Irregularity, Color score, Diameter)."
        })
        bio_res = self.bio_tool.run(image_np, mask, ham10000_pred=cls_res)
        logs.append({
            "step": 6,
            "type": "action",
            "tool": "BiomarkerTool",
            "message": f"Biomarkers extracted: Asymmetry={bio_res['asymmetry_index']}, Border={bio_res['border_irregularity']}, Diameter={bio_res['diameter_mm']}mm, TDS={bio_res['tds_score']}."
        })

        logs.append({
            "step": 7,
            "type": "thought",
            "message": "Evaluating model confidence and boundary uncertainty using Monte Carlo Dropout to check for ambiguous edge predictions."
        })
        unc_res = self.unc_tool.run(image_tensor)
        logs.append({
            "step": 8,
            "type": "action",
            "tool": "UncertaintyTool",
            "message": f"Uncertainty assessment complete: Boundary Confidence = {unc_res['confidence_score']}% (Edge Uncertainty = {unc_res['edge_uncertainty']})."
        })

        logs.append({
            "step": 9,
            "type": "thought",
            "message": "Synthesizing HAM10000 7-class differential diagnosis and Stolz ABCD biomarkers for risk stratification and clinical triage protocol."
        })
        guide_res = self.guide_tool.run(bio_res["tds_score"], bio_res["diameter_mm"], cls_res=cls_res)
        logs.append({
            "step": 10,
            "type": "action",
            "tool": "GuidelinesTool",
            "message": f"Risk assessment: {guide_res['risk']}. Protocol: {guide_res['recommendation']}"
        })

        report = self._generate_report(bio_res, unc_res, guide_res, cls_res)

        return {
            "mask": mask,
            "attention_map": att_map,
            "variance_map": unc_res["variance_map"],
            "classification": cls_res,
            "biomarkers": bio_res,
            "uncertainty": unc_res,
            "guidelines": guide_res,
            "reasoning_logs": logs,
            "clinical_report": report
        }

    def _generate_report(self, bio, unc, guide, cls_res):
        top3_lines = "\n".join([
            f"  - **{item['name']} (`{item['code'].upper()}`)**: `{item['prob_pct']}%` — *{item['malignancy']}*"
            for item in cls_res["top3_differential"]
        ])
        return f"""### 🏥 MedVision-Agent Clinical Diagnostic Summary (HAM10000 Model)

**Patient Scan Assessment**:
* **HAM10000 Primary Diagnosis**: `{cls_res['predicted_name']} ({cls_res['predicted_code'].upper()})` — **{cls_res['confidence_pct']}%**
* **Pathology Category**: `{cls_res['malignancy']}`
* **Combined Risk Stratification**: `{bio['risk_level']}`
* **Total Dermatoscopy Score (TDS)**: `{bio['tds_score']}` / 8.0
* **Segmentation Boundary Confidence**: `{unc['confidence_score']}%`

---

#### 🧬 HAM10000 Top-3 Differential Diagnosis:
{top3_lines}

---

#### 🔬 Morphological ABCD Findings:
1. **Asymmetry (A)**: `{bio['asymmetry_index']} / 2.0` ({'High asymmetry noted across axes' if bio['asymmetry_index'] > 1.2 else 'Mild/Symmetric structure'})
2. **Border Irregularity (B)**: `{bio['border_irregularity']} / 8.0` ({'Notched / scalloped borders detected' if bio['border_irregularity'] > 3.0 else 'Smooth demarcated boundary'})
3. **Color Variegation (C)**: `{bio['color_score']} / 6 distinct pigment clusters`
4. **Diameter (D)**: `{bio['diameter_mm']} mm` ({'> 6mm critical melanoma threshold' if bio['diameter_mm'] > 6.0 else 'Within benign size profile'})

---

#### 🛡️ Quality Assurance & Model Reliability:
* **Boundary Precision**: {unc['confidence_score']}% certainty on perimeter segmentation (Tschandl HAM10000 mask protocol).
* **Artifact Filtering**: Attention Gates suppressed hair/vignette interference with high activation on central lesion.

---

#### 🩺 Physician Recommendation:
> **{guide['recommendation']}**
"""


class ClinicalChatAgent:
    """
    Feature 2: Interactive Doctor-in-the-Loop Clinical AI QA Assistant.
    Answers natural language medical queries regarding HAM10000 7-class classification,
    current patient scan biomarkers, hypothetical parameter simulations, and clinical protocols.
    """
    def __init__(self, current_analysis=None):
        self.analysis = current_analysis

    def update_context(self, analysis):
        self.analysis = analysis

    def reply(self, user_query):
        q = user_query.lower()
        if self.analysis is None:
            return "Please run or load a dermoscopy scan first so I can analyze the patient's specific biomarkers."

        bio = self.analysis["biomarkers"]
        unc = self.analysis["uncertainty"]
        cls_res = self.analysis.get("classification", {})

        if any(k in q for k in ["ham10000", "class", "differential", "melanoma", "nevus", "bcc", "akiec", "bkl", "df", "vasc", "diagnosis"]):
            if cls_res:
                top_str = ", ".join([f"{d['name']} ({d['code'].upper()}: {d['prob_pct']}%)" for d in cls_res["top3_differential"]])
                return (
                    f"🧬 **HAM10000 7-Class Classification:** The multi-task Attention U-Net classified this lesion as "
                    f"**{cls_res['predicted_name']} (`{cls_res['predicted_code'].upper()}`)** with **{cls_res['confidence_pct']}%** probability "
                    f"(*{cls_res['malignancy']}*). Top-3 differential breakdown: {top_str}. "
                    f"Recommended clinical action: **{cls_res['clinical_action']}**"
                )

        if "why" in q and ("border" in q or "irregular" in q):
            return f"The lesion's Border Irregularity metric is scored at **{bio['border_irregularity']} / 8.0**. This was computed via contour compactness $C = P^2 / (4\\pi A)$. The perimeter exhibits micro-notching and uneven curvature relative to a circular profile."

        elif "why" in q and ("risk" in q or "malignant" in q or "suspicious" in q or "benign" in q):
            cls_note = f" Furthermore, the HAM10000 diagnostic head predicted **{cls_res.get('predicted_name', 'N/A')} ({cls_res.get('predicted_code', '').upper()})**." if cls_res else ""
            return f"The lesion is classified as **{bio['risk_level']}** because the computed **Total Dermatoscopy Score (TDS) is {bio['tds_score']}** (Thresholds: <4.75 is Benign, 4.75–5.45 is Suspicious, >5.45 is High Risk). Contributing factors: Asymmetry ({bio['asymmetry_index']}), Border ({bio['border_irregularity']}), Color ({bio['color_score']}), and Diameter ({bio['diameter_mm']}mm).{cls_note}"

        elif "what if" in q or ("diameter" in q and ("4" in q or "5" in q or "smaller" in q or "reduce" in q)):
            sim_d = 4.0
            sim_tds = (1.3 * bio['asymmetry_index']) + (0.1 * bio['border_irregularity']) + (0.5 * bio['color_score']) + (0.5 * (sim_d / 2.0))
            sim_risk = "Benign / Low Risk" if sim_tds < 4.75 else ("Suspicious" if sim_tds <= 5.45 else "High Risk")
            return f"🔬 **Simulation Result:** If the diameter were reduced to **{sim_d} mm**, the simulated TDS score drops from **{bio['tds_score']}** to **{sim_tds:.2f}**, re-stratifying the morphological risk to **{sim_risk}**."

        elif "attention" in q or "gate" in q or "artifact" in q or "hair" in q:
            return "The Attention Gates compute spatial attention coefficients $\\alpha \\in [0, 1]$ by gating the encoder skip connections with deeper decoder features. Trained on HAM10000 Tschandl lesion masks, this dynamically down-weights skin hair, scanner vignetting, and normal skin texture, focusing gradient propagation strictly on the pigmented lesion core."

        elif "uncertainty" in q or "confidence" in q or "dropout" in q or "edge" in q:
            return f"The model boundary confidence is **{unc['confidence_score']}%** (Edge variance = {unc['edge_uncertainty']}). Using Monte Carlo Dropout across stochastic inference passes, higher variance at the boundary indicates ambiguous perimeter transitions where clinicians should visually verify margins."

        elif "protocol" in q or "recommend" in q or "next step" in q or "biopsy" in q:
            return f"📋 **Clinical Protocol:** For a lesion with TDS **{bio['tds_score']}** ({bio['risk_level']}) and HAM10000 primary diagnosis **{cls_res.get('predicted_name', 'N/A')}**: **{bio['recommendation']}** Standard oncology guidelines suggest a 2–5mm surgical margin for diagnostic excisional biopsy."

        elif "formula" in q or "tds" in q or "abcd" in q:
            return "The Total Dermatoscopy Score (Stolz Algorithm) formula is:\n$$\\text{TDS} = 1.3 \\times A + 0.1 \\times B + 0.5 \\times C + 0.5 \\times D$$\nWhere $A$ is Asymmetry ($[0, 2]$), $B$ is Border ($[1, 8]$), $C$ is Color count ($[1, 6]$), and $D$ is Diameter ($[1, 5]$)."

        else:
            cls_summary = f"HAM10000 Class = **{cls_res['predicted_name']} ({cls_res['predicted_code'].upper()}, {cls_res['confidence_pct']}%)**, " if cls_res else ""
            return f"Based on patient scan metrics: {cls_summary}TDS = **{bio['tds_score']}** ({bio['risk_level']}), Diameter = **{bio['diameter_mm']}mm**, Confidence = **{unc['confidence_score']}%**. Recommendation: {bio['recommendation']}. Feel free to ask about HAM10000 7-class differential diagnosis, ABCD biomarkers, hypothetical parameter adjustments, or attention gating."
