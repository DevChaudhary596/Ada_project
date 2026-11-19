import os
import json
import time
import argparse
import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from dataset import (
    HAM10000_CLASSES,
    HAM10000_CLASS_INFO,
    get_default_data_dir,
    download_ham10000_dataset,
    build_local_manifest,
    create_ham10000_dataloaders
)
from model import (
    AttentionUNet,
    UNet,
    MultiTaskHAM10000Loss,
    compute_segmentation_metrics
)


def evaluate_model(model, val_loader, criterion, device):
    """Evaluates segmentation (Dice, IoU, Precision, Recall) and 7-class classification metrics on HAM10000 val split."""
    model.eval()
    val_loss_total = 0.0
    val_loss_seg = 0.0
    val_loss_cls = 0.0

    dice_list, iou_list, prec_list, rec_list = [], [], [], []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, masks, labels, _ in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            seg_logits, cls_logits = model(images, return_classification=True)
            loss_tot, loss_s, loss_c = criterion(seg_logits, cls_logits, masks, labels)

            val_loss_total += loss_tot.item()
            val_loss_seg += loss_s.item()
            val_loss_cls += loss_c.item()

            seg_m = compute_segmentation_metrics(seg_logits, masks)
            dice_list.append(seg_m["dice"])
            iou_list.append(seg_m["iou"])
            prec_list.append(seg_m["precision"])
            rec_list.append(seg_m["recall"])

            preds = torch.argmax(cls_logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    n_batches = max(1, len(val_loader))
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    cls_acc = float(np.mean(all_preds_np == all_labels_np)) if len(all_labels_np) > 0 else 0.0
    cls_f1 = float(f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0)) if len(all_labels_np) > 0 else 0.0
    cm = confusion_matrix(all_labels_np, all_preds_np, labels=list(range(len(HAM10000_CLASSES)))).tolist()

    per_class_acc = {}
    for idx, cls_code in enumerate(HAM10000_CLASSES):
        cls_mask = (all_labels_np == idx)
        if np.sum(cls_mask) > 0:
            per_class_acc[cls_code] = round(float(np.mean(all_preds_np[cls_mask] == idx)) * 100.0, 1)
        else:
            per_class_acc[cls_code] = 0.0

    return {
        "loss_total": round(val_loss_total / n_batches, 4),
        "loss_seg": round(val_loss_seg / n_batches, 4),
        "loss_cls": round(val_loss_cls / n_batches, 4),
        "dice": round(float(np.mean(dice_list)), 4),
        "iou": round(float(np.mean(iou_list)), 4),
        "precision": round(float(np.mean(prec_list)), 4),
        "recall": round(float(np.mean(rec_list)), 4),
        "cls_acc": round(cls_acc, 4),
        "cls_f1": round(cls_f1, 4),
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm
    }


def run_ham10000_training(
    data_dir=None,
    img_size=(256, 256),
    epochs=12,
    batch_size=8,
    lr=1e-3,
    samples_per_class=25,
    full_download=False,
    run_ablation=True,
    save_dir=None,
    progress_callback=None
):
    """
    End-to-end HAM10000 training pipeline for Attention U-Net (and optional Baseline U-Net ablation).
    Downloads dataset if needed, trains the model, saves best_attention_unet.pth and training_history.json.
    """
    if data_dir is None:
        data_dir = get_default_data_dir()
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _notify(msg, pct):
        print(f"[Train] {msg}")
        if progress_callback is not None:
            progress_callback(msg, pct)

    # 1. Verify or download HAM10000 dataset
    manifest_df = build_local_manifest(data_dir)
    min_expected = samples_per_class * len(HAM10000_CLASSES)
    if len(manifest_df) < min(35, min_expected):
        _notify("Fetching HAM10000 dermoscopy images, Tschandl segmentation masks, and metadata...", 0.05)
        manifest_df = download_ham10000_dataset(
            data_dir=data_dir,
            samples_per_class=samples_per_class,
            full_download=full_download
        )

    _notify(f"Initializing HAM10000 DataLoaders ({len(manifest_df)} samples across 7 classes) on {device}...", 0.10)
    train_loader, val_loader, class_weights, train_df, val_df = create_ham10000_dataloaders(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        val_ratio=0.2
    )

    class_weights = class_weights.to(device)
    criterion = MultiTaskHAM10000Loss(class_weights=class_weights, seg_weight=1.0, cls_weight=0.55)

    model = AttentionUNet(
        in_channels=3,
        out_channels=1,
        num_classes=len(HAM10000_CLASSES),
        features=[32, 64, 128, 256],
        dropout_rate=0.15
    ).to(device)

    ckpt_path = os.path.join(save_dir, "best_attention_unet.pth")
    if os.path.exists(ckpt_path):
        try:
            prev_state = torch.load(ckpt_path, map_location=device)
            sd = prev_state.get("model_state_dict", prev_state)
            model.load_state_dict(sd, strict=False)
            _notify("Loaded existing checkpoint weights for continual HAM10000 fine-tuning.", 0.12)
        except Exception as e:
            print(f"[Train] Starting from fresh weights ({e})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.05)

    history = {
        "dataset": "HAM10000 (Human Against Machine with 10000 training images)",
        "classes": HAM10000_CLASSES,
        "total_samples": len(manifest_df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "img_size": list(img_size),
        "device": str(device),
        "epochs_trained": epochs,
        "class_distribution": train_df["dx"].value_counts().to_dict(),
        "epochs": [],
        "best_metrics": {},
        "ablation": {}
    }

    best_score = -1.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_tot = 0.0
        train_loss_seg = 0.0
        train_loss_cls = 0.0
        train_dice_list = []
        train_correct = 0
        train_count = 0

        for images, masks, labels, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            seg_logits, cls_logits = model(images, return_classification=True)
            loss_tot, loss_s, loss_c = criterion(seg_logits, cls_logits, masks, labels)

            loss_tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss_tot += loss_tot.item()
            train_loss_seg += loss_s.item()
            train_loss_cls += loss_c.item()

            seg_m = compute_segmentation_metrics(seg_logits, masks)
            train_dice_list.append(seg_m["dice"])

            preds = torch.argmax(cls_logits, dim=1)
            train_correct += int((preds == labels).sum().item())
            train_count += labels.size(0)

        scheduler.step()

        n_train_batches = max(1, len(train_loader))
        tr_loss = round(train_loss_tot / n_train_batches, 4)
        tr_dice = round(float(np.mean(train_dice_list)), 4)
        tr_acc = round(train_correct / max(1, train_count), 4)

        val_metrics = evaluate_model(model, val_loader, criterion, device)

        epoch_record = {
            "epoch": epoch,
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "train_loss": tr_loss,
            "train_dice": tr_dice,
            "train_cls_acc": tr_acc,
            "val_loss": val_metrics["loss_total"],
            "val_seg_loss": val_metrics["loss_seg"],
            "val_cls_loss": val_metrics["loss_cls"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_cls_acc": val_metrics["cls_acc"],
            "val_cls_f1": val_metrics["cls_f1"]
        }
        history["epochs"].append(epoch_record)

        # Combined model selection criterion (Segmentation Dice + Classification Accuracy)
        composite_score = 0.65 * val_metrics["dice"] + 0.35 * val_metrics["cls_acc"]
        if composite_score >= best_score:
            best_score = composite_score
            history["best_metrics"] = {
                "best_epoch": epoch,
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_cls_acc": val_metrics["cls_acc"],
                "val_cls_f1": val_metrics["cls_f1"],
                "per_class_acc": val_metrics["per_class_acc"],
                "confusion_matrix": val_metrics["confusion_matrix"]
            }
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_metrics": history["best_metrics"],
                "classes": HAM10000_CLASSES,
                "dataset": "HAM10000",
                "img_size": list(img_size)
            }, ckpt_path)

        pct = 0.15 + 0.70 * (epoch / max(1, epochs))
        _notify(
            f"Epoch [{epoch}/{epochs}] | Train Loss: {tr_loss:.4f} (Dice: {tr_dice:.3f}, Acc: {tr_acc*100:.1f}%) | "
            f"Val Dice: {val_metrics['dice']:.4f} | Val IoU: {val_metrics['iou']:.4f} | Val 7-Class Acc: {val_metrics['cls_acc']*100:.1f}%",
            pct
        )

    # Optional: Run Baseline U-Net ablation on the exact same HAM10000 split for real comparison
    if run_ablation:
        _notify("Running Baseline U-Net ablation study on HAM10000 split...", 0.88)
        baseline_model = UNet(
            in_channels=3,
            out_channels=1,
            num_classes=len(HAM10000_CLASSES),
            features=[32, 64, 128, 256]
        ).to(device)
        base_opt = torch.optim.AdamW(baseline_model.parameters(), lr=lr, weight_decay=1e-4)
        ablation_epochs = max(3, epochs // 2)
        for _ in range(ablation_epochs):
            baseline_model.train()
            for images, masks, labels, _ in train_loader:
                images, masks, labels = images.to(device), masks.to(device), labels.to(device)
                base_opt.zero_grad()
                s_log, c_log = baseline_model(images, return_classification=True)
                l_tot, _, _ = criterion(s_log, c_log, masks, labels)
                l_tot.backward()
                base_opt.step()

        base_val = evaluate_model(baseline_model, val_loader, criterion, device)
        history["ablation"] = {
            "baseline_unet": {
                "val_dice": base_val["dice"],
                "val_iou": base_val["iou"],
                "val_precision": base_val["precision"],
                "val_recall": base_val["recall"],
                "val_cls_acc": base_val["cls_acc"],
                "val_cls_f1": base_val["cls_f1"],
                "params_m": round(sum(p.numel() for p in baseline_model.parameters()) / 1e6, 2)
            },
            "attention_unet": {
                "val_dice": history["best_metrics"]["val_dice"],
                "val_iou": history["best_metrics"]["val_iou"],
                "val_precision": history["best_metrics"]["val_precision"],
                "val_recall": history["best_metrics"]["val_recall"],
                "val_cls_acc": history["best_metrics"]["val_cls_acc"],
                "val_cls_f1": history["best_metrics"]["val_cls_f1"],
                "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
            }
        }
        torch.save({
            "model_state_dict": baseline_model.state_dict(),
            "best_metrics": base_val
        }, os.path.join(save_dir, "best_baseline_unet.pth"))

    elapsed = round(time.time() - start_time, 1)
    history["training_time_sec"] = elapsed

    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    _notify(
        f"Training complete in {elapsed}s! Best Val Dice: {history['best_metrics']['val_dice']:.4f}, "
        f"Best Val IoU: {history['best_metrics']['val_iou']:.4f}, "
        f"Best 7-Class Acc: {history['best_metrics']['val_cls_acc']*100:.1f}%. Saved to {ckpt_path}",
        1.0
    )
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Task Attention U-Net on HAM10000 Dataset")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to local HAM10000 directory")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--img-size", type=int, default=256, help="Input spatial resolution (H=W)")
    parser.add_argument("--samples-per-class", type=int, default=25, help="Samples per class for fast stratified download")
    parser.add_argument("--full-download", action="store_true", help="Download all 10,015 HAM10000 images (~2.6 GB)")
    parser.add_argument("--no-ablation", action="store_true", help="Skip Baseline U-Net ablation training")
    args = parser.parse_args()

    run_ham10000_training(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        samples_per_class=args.samples_per_class,
        full_download=args.full_download,
        run_ablation=not args.no_ablation
    )
