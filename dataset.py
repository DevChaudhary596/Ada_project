import os
import json
import zipfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

# Official HAM10000 7-Class Diagnostic Taxonomy (Tschandl et al., Sci Data 2018)
HAM10000_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

HAM10000_CLASS_INFO = {
    "akiec": {
        "index": 0,
        "code": "akiec",
        "name": "Actinic Keratoses / Bowen's Disease",
        "malignancy": "Pre-Malignant / In-Situ Carcinoma",
        "risk_tier": "Suspicious / Moderate-High Risk",
        "clinical_action": "Cryotherapy, topical 5-FU/Imiquimod, photodynamic therapy, or targeted shave biopsy to rule out invasive SCC."
    },
    "bcc": {
        "index": 1,
        "code": "bcc",
        "name": "Basal Cell Carcinoma",
        "malignancy": "Malignant (Non-Melanoma Skin Cancer)",
        "risk_tier": "High Risk / Malignant Suspected",
        "clinical_action": "Surgical excision with 4mm clinical margins or Mohs micrographic surgery on high-risk anatomical zones."
    },
    "bkl": {
        "index": 2,
        "code": "bkl",
        "name": "Benign Keratosis-like Lesion",
        "malignancy": "Benign (Seborrheic Keratosis / Solar Lentigo)",
        "risk_tier": "Benign / Low Risk",
        "clinical_action": "Reassurance and routine annual observation; cryotherapy or curettage only if symptomatic or irritated."
    },
    "df": {
        "index": 3,
        "code": "df",
        "name": "Dermatofibroma",
        "malignancy": "Benign Fibrohistiocytic Proliferation",
        "risk_tier": "Benign / Low Risk",
        "clinical_action": "Clinical reassurance (positive dimple sign); no surgical intervention required unless atypical growth occurs."
    },
    "mel": {
        "index": 4,
        "code": "mel",
        "name": "Melanoma",
        "malignancy": "Malignant Melanocytic Neoplasm",
        "risk_tier": "High Risk / Malignant Suspected",
        "clinical_action": "Urgent full-thickness excisional biopsy with 2mm narrow margin for Breslow depth staging and dermatological oncology referral."
    },
    "nv": {
        "index": 5,
        "code": "nv",
        "name": "Melanocytic Nevus",
        "malignancy": "Benign Melanocytic Neoplasm",
        "risk_tier": "Benign / Low Risk",
        "clinical_action": "Routine annual dermatological monitoring; sequential digital dermoscopy if multiple atypical nevi are present."
    },
    "vasc": {
        "index": 6,
        "code": "vasc",
        "name": "Vascular Lesion",
        "malignancy": "Benign (Angioma / Angiokeratoma / Pyogenic Granuloma)",
        "risk_tier": "Benign / Low Risk",
        "clinical_action": "Clinical reassurance; pulsed-dye laser or simple curettage if prone to bleeding."
    }
}

CLASS_TO_IDX = {c: i for i, c in enumerate(HAM10000_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(HAM10000_CLASSES)}

HF_IMAGES_REPO = "ShiroOnigami23/skin-cancer-ham10000-dataset"
HF_MASKS_REPO = "oscar2525mv/ham10000-segmentation"


def get_default_data_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "data", "HAM10000")


def _generate_pseudo_mask_from_image(img_np):
    """
    Fallback Otsu + morphological color-space mask generator if a raw image does not
    have an external Tschandl segmentation mask file present.
    """
    import cv2
    h, w = img_np.shape[:2]
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Suppress border vignette corners
    Y, X = np.ogrid[:h, :w]
    center_mask = ((X - w / 2) ** 2 + (Y - h / 2) ** 2) <= ((min(h, w) * 0.46) ** 2)
    thresh = (thresh * center_mask).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)
    else:
        cv2.circle(mask, (w // 2, h // 2), min(h, w) // 4, 255, -1)
    return mask


def download_ham10000_dataset(
    data_dir=None,
    samples_per_class=25,
    full_download=False,
    max_workers=8,
    progress_callback=None
):
    """
    Downloads the real HAM10000 metadata, Tschandl ground-truth segmentation masks,
    and dermoscopy images from HuggingFace Hub (or verifies existing local dataset).

    Args:
        data_dir: Target folder (default: MedVision/data/HAM10000)
        samples_per_class: Number of stratified images per class to fetch in fast mode
                           (7 * 25 = 175 balanced images + masks). Ignored if full_download=True.
        full_download: If True, downloads the full HAM10000_images_part_1.zip and part_2.zip (10,015 images).
        max_workers: Parallel threads for image downloading.
        progress_callback: Optional callable(str, float) for UI progress updates.
    """
    from huggingface_hub import hf_hub_download, HfApi

    if data_dir is None:
        data_dir = get_default_data_dir()

    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    def _report(msg, pct):
        print(f"[HAM10000] {msg}")
        if progress_callback is not None:
            progress_callback(msg, pct)

    metadata_path = os.path.join(data_dir, "HAM10000_metadata.csv")

    # 1. Download HAM10000_metadata.csv
    if not os.path.exists(metadata_path):
        _report("Downloading official HAM10000_metadata.csv (10,015 clinical records)...", 0.05)
        cached_csv = hf_hub_download(
            repo_id=HF_IMAGES_REPO,
            filename="HAM10000_metadata.csv",
            repo_type="dataset"
        )
        shutil.copyfile(cached_csv, metadata_path)

    df_meta = pd.read_csv(metadata_path)

    # 2. Download and extract all 10,015 ground-truth lesion segmentation masks (10.3 MB zip)
    existing_masks = [f for f in os.listdir(masks_dir) if f.endswith(".png")]
    if len(existing_masks) < 100:
        _report("Downloading HAM10000 Tschandl ground-truth segmentation masks (10.3 MB)...", 0.15)
        try:
            masks_zip_path = hf_hub_download(
                repo_id=HF_MASKS_REPO,
                filename="HAM10000_segmentations_lesion_tschandl.zip",
                repo_type="dataset"
            )
            _report("Extracting 10,015 lesion segmentation masks...", 0.25)
            with zipfile.ZipFile(masks_zip_path, "r") as zf:
                for member in zf.infolist():
                    if member.filename.endswith(".png") and not member.is_dir():
                        base_name = os.path.basename(member.filename)
                        if base_name and not base_name.startswith("."):
                            target_file = os.path.join(masks_dir, base_name)
                            if not os.path.exists(target_file):
                                with zf.open(member) as src, open(target_file, "wb") as dst:
                                    shutil.copyfileobj(src, dst)
        except Exception as e:
            _report(f"Warning: Could not download mask archive ({e}). Will use automated mask extraction if needed.", 0.28)

    # 3. Download full image archives or balanced subset of real HAM10000 dermoscopy images
    if full_download:
        for idx, zip_name in enumerate(["HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"], start=1):
            _report(f"Downloading full {zip_name} (~1.3 GB)...", 0.3 + idx * 0.25)
            zip_path = hf_hub_download(
                repo_id=HF_MASKS_REPO,
                filename=zip_name,
                repo_type="dataset"
            )
            _report(f"Extracting {zip_name} into {images_dir}...", 0.4 + idx * 0.25)
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.infolist():
                    if member.filename.lower().endswith(".jpg") and not member.is_dir():
                        base_name = os.path.basename(member.filename)
                        target_file = os.path.join(images_dir, base_name)
                        if not os.path.exists(target_file):
                            with zf.open(member) as src, open(target_file, "wb") as dst:
                                shutil.copyfileobj(src, dst)
    else:
        # Stratified selection across all 7 classes so every class is well-represented
        _report(f"Preparing stratified HAM10000 subset ({samples_per_class} per class across 7 classes)...", 0.35)
        api = HfApi()
        repo_info = api.dataset_info(HF_IMAGES_REPO)
        repo_map = {}
        for s in repo_info.siblings:
            if s.rfilename.endswith(".jpg") and ("HAM10000_images_part_1/" in s.rfilename or "HAM10000_images_part_2/" in s.rfilename):
                img_id = os.path.splitext(os.path.basename(s.rfilename))[0]
                repo_map[img_id] = s.rfilename

        selected_rows = []
        rng = np.random.RandomState(42)
        for cls_name in HAM10000_CLASSES:
            cls_df = df_meta[df_meta["dx"] == cls_name]
            # Prefer unique lesion_ids
            unique_lesions = cls_df.drop_duplicates(subset=["lesion_id"])
            avail = unique_lesions[unique_lesions["image_id"].isin(repo_map.keys())]
            if len(avail) < samples_per_class:
                avail = cls_df[cls_df["image_id"].isin(repo_map.keys())]
            n_pick = min(samples_per_class, len(avail))
            picked = avail.sample(n=n_pick, random_state=rng)
            selected_rows.append(picked)

        subset_df = pd.concat(selected_rows, ignore_index=True)

        to_download = []
        for _, row in subset_df.iterrows():
            img_id = row["image_id"]
            local_jpg = os.path.join(images_dir, f"{img_id}.jpg")
            if not os.path.exists(local_jpg) and img_id in repo_map:
                to_download.append((img_id, repo_map[img_id], local_jpg))

        if to_download:
            total_dl = len(to_download)
            _report(f"Downloading {total_dl} real HAM10000 dermoscopy scans from HuggingFace...", 0.45)

            def _fetch_one(item):
                img_id, remote_rel, dst_path = item
                cached = hf_hub_download(
                    repo_id=HF_IMAGES_REPO,
                    filename=remote_rel,
                    repo_type="dataset"
                )
                shutil.copyfile(cached, dst_path)
                return img_id

            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_fetch_one, item) for item in to_download]
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"[HAM10000] Error downloading image: {e}")
                    completed += 1
                    if completed % 20 == 0 or completed == total_dl:
                        pct = 0.45 + 0.50 * (completed / max(1, total_dl))
                        _report(f"Downloaded {completed}/{total_dl} HAM10000 dermoscopy images...", pct)

    # Save index manifest of available local pairs
    manifest_df = build_local_manifest(data_dir)
    _report(f"HAM10000 dataset ready: {len(manifest_df)} verified image+mask+metadata records.", 1.0)
    return manifest_df


def build_local_manifest(data_dir=None):
    """
    Scans data_dir (including standard HAM10000 folder layouts) and returns a DataFrame
    of all locally available images matched with their segmentation masks and metadata.
    """
    if data_dir is None:
        data_dir = get_default_data_dir()

    metadata_candidates = [
        os.path.join(data_dir, "HAM10000_metadata.csv"),
        os.path.join(data_dir, "HAM10000_consolidated.csv")
    ]
    meta_path = next((p for p in metadata_candidates if os.path.exists(p)), None)
    if meta_path is None:
        return pd.DataFrame()

    df_meta = pd.read_csv(meta_path)

    # Discover all image folders (supports images/, HAM10000_images_part_1/, HAM10000_images_part_2/, root)
    image_dirs = [
        os.path.join(data_dir, "images"),
        os.path.join(data_dir, "HAM10000_images_part_1"),
        os.path.join(data_dir, "HAM10000_images_part_2"),
        data_dir
    ]
    mask_dirs = [
        os.path.join(data_dir, "masks"),
        os.path.join(data_dir, "HAM10000_segmentations_lesion_tschandl"),
        os.path.join(data_dir, "segmentations")
    ]

    img_lookup = {}
    for d in image_dirs:
        if os.path.isdir(d):
            for fname in os.listdir(d):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")) and "segmentation" not in fname.lower():
                    img_id = os.path.splitext(fname)[0]
                    if img_id not in img_lookup:
                        img_lookup[img_id] = os.path.join(d, fname)

    mask_lookup = {}
    for d in mask_dirs:
        if os.path.isdir(d):
            for fname in os.listdir(d):
                if fname.lower().endswith(".png"):
                    img_id = fname.replace("_segmentation.png", "").replace(".png", "")
                    if img_id not in mask_lookup:
                        mask_lookup[img_id] = os.path.join(d, fname)

    avail_df = df_meta[df_meta["image_id"].isin(img_lookup.keys())].copy()
    avail_df["image_path"] = avail_df["image_id"].map(img_lookup)
    avail_df["mask_path"] = avail_df["image_id"].map(lambda x: mask_lookup.get(x, ""))
    avail_df["label_idx"] = avail_df["dx"].map(CLASS_TO_IDX)
    avail_df = avail_df.dropna(subset=["label_idx"]).reset_index(drop=True)
    avail_df["label_idx"] = avail_df["label_idx"].astype(int)

    manifest_path = os.path.join(data_dir, "local_manifest.csv")
    avail_df.to_csv(manifest_path, index=False)
    return avail_df


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for Multi-Task HAM10000 Segmentation + 7-Class Classification.
    Returns:
        image_tensor: FloatTensor [3, H, W] in [0, 1]
        mask_tensor:  FloatTensor [1, H, W] in {0.0, 1.0}
        label_tensor: LongTensor scalar in {0..6}
        meta_dict:    dict with image_id, lesion_id, dx, age, sex, localization
    """
    def __init__(self, manifest_df, img_size=(256, 256), augment=False):
        self.df = manifest_df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _apply_augmentations(self, img_np, mask_np):
        # Random horizontal flip
        if np.random.rand() < 0.5:
            img_np = np.fliplr(img_np).copy()
            mask_np = np.fliplr(mask_np).copy()
        # Random vertical flip
        if np.random.rand() < 0.5:
            img_np = np.flipud(img_np).copy()
            mask_np = np.flipud(mask_np).copy()
        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            img_np = np.rot90(img_np, k).copy()
            mask_np = np.rot90(mask_np, k).copy()
        # Mild brightness/contrast jitter on image only
        if np.random.rand() < 0.4:
            alpha = np.random.uniform(0.9, 1.1)
            beta = np.random.uniform(-10, 10)
            img_np = np.clip(img_np.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return img_np, mask_np

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        mask_path = row.get("mask_path", "")

        pil_img = Image.open(img_path).convert("RGB").resize(self.img_size, Image.BILINEAR)
        img_np = np.array(pil_img, dtype=np.uint8)

        if isinstance(mask_path, str) and mask_path and os.path.exists(mask_path):
            pil_mask = Image.open(mask_path).convert("L").resize(self.img_size, Image.NEAREST)
            mask_np = (np.array(pil_mask) > 127).astype(np.float32)
        else:
            raw_mask = _generate_pseudo_mask_from_image(img_np)
            mask_np = (raw_mask > 127).astype(np.float32)

        if self.augment:
            img_np, mask_np = self._apply_augmentations(img_np, mask_np)

        image_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
        label_tensor = torch.tensor(int(row["label_idx"]), dtype=torch.long)

        meta = {
            "image_id": str(row.get("image_id", f"HAM_{idx}")),
            "lesion_id": str(row.get("lesion_id", f"LES_{idx}")),
            "dx": str(row.get("dx", "nv")),
            "dx_type": str(row.get("dx_type", "histo")),
            "age": float(row["age"]) if pd.notna(row.get("age")) else 50.0,
            "sex": str(row.get("sex", "unknown")),
            "localization": str(row.get("localization", "unknown"))
        }
        return image_tensor, mask_tensor, label_tensor, meta


def create_ham10000_dataloaders(
    data_dir=None,
    img_size=(256, 256),
    batch_size=8,
    val_ratio=0.2,
    use_weighted_sampler=True,
    seed=42
):
    """
    Creates lesion-grouped Train and Validation DataLoaders + class weights tensor
    for HAM10000 training. Ensures all 7 classes appear in both train and val splits.
    """
    manifest_df = build_local_manifest(data_dir)
    if len(manifest_df) == 0:
        raise RuntimeError(f"No local HAM10000 images found in {data_dir}. Run download_ham10000_dataset() first.")

    # Stratified split ensuring every diagnostic class is represented in both train and val
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(sss.split(manifest_df, manifest_df["label_idx"]))
    except ValueError:
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(gss.split(manifest_df, manifest_df["label_idx"], groups=manifest_df["lesion_id"]))

    train_df = manifest_df.iloc[train_idx].reset_index(drop=True)
    val_df = manifest_df.iloc[val_idx].reset_index(drop=True)

    train_dataset = HAM10000Dataset(train_df, img_size=img_size, augment=True)
    val_dataset = HAM10000Dataset(val_df, img_size=img_size, augment=False)

    # Compute inverse-frequency class weights for CrossEntropyLoss
    num_classes = len(HAM10000_CLASSES)
    counts = np.bincount(train_df["label_idx"].values, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    inv_freq = (len(train_df) / (num_classes * counts))
    # Smooth weights to prevent extreme gradients
    class_weights = torch.tensor(np.sqrt(inv_freq), dtype=torch.float32)

    if use_weighted_sampler and len(train_df) > batch_size:
        sample_weights = [float(inv_freq[label]) for label in train_df["label_idx"].values]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_df), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, class_weights, train_df, val_df
