import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution -> BatchNorm -> ReLU) * 2 with optional Dropout"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate (Oktay et al., 2018)
    Suppresses irrelevant background features and noise in skip connections.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha, alpha


class AttentionUNet(nn.Module):
    """
    Multi-Task Attention U-Net Architecture (Oktay et al., 2018) trained on HAM10000.
    Simultaneously performs:
      1. Pixel-wise lesion segmentation via Attention-Gated skip connections.
      2. 7-Class HAM10000 diagnostic classification (akiec, bcc, bkl, df, mel, nv, vasc)
         using attention-weighted multi-scale encoder & bottleneck features.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        num_classes=7,
        features=[32, 64, 128, 256],
        dropout_rate=0.15
    ):
        super(AttentionUNet, self).__init__()
        self.num_classes = num_classes
        self.features = features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = DoubleConv(in_channels, features[0], dropout_rate=0.0)
        self.enc2 = DoubleConv(features[0], features[1], dropout_rate=0.0)
        self.enc3 = DoubleConv(features[1], features[2], dropout_rate=dropout_rate)
        self.enc4 = DoubleConv(features[2], features[3], dropout_rate=dropout_rate)

        self.bottleneck = DoubleConv(features[3], features[3] * 2, dropout_rate=dropout_rate)

        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.ag4 = AttentionGate(F_g=features[3], F_l=features[3], F_int=features[3] // 2)
        self.dec4 = DoubleConv(features[3] * 2, features[3], dropout_rate=dropout_rate)

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.ag3 = AttentionGate(F_g=features[2], F_l=features[2], F_int=features[2] // 2)
        self.dec3 = DoubleConv(features[2] * 2, features[2], dropout_rate=dropout_rate)

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.ag2 = AttentionGate(F_g=features[1], F_l=features[1], F_int=features[1] // 2)
        self.dec2 = DoubleConv(features[1] * 2, features[1], dropout_rate=0.0)

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.ag1 = AttentionGate(F_g=features[0], F_l=features[0], F_int=features[0] // 2)
        self.dec1 = DoubleConv(features[0] * 2, features[0], dropout_rate=0.0)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Multi-scale HAM10000 7-class diagnostic classification head
        # Combines pooled bottleneck (2*F3), attention-gated e4 (F3), and attention-gated e3 (F2)
        cls_in_dim = (features[3] * 2) * 2 + features[3] + features[2]
        self.classifier = nn.Sequential(
            nn.Linear(cls_in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(128, num_classes)
        )

    def _compute_cls_logits(self, b, x4, x3):
        b_avg = F.adaptive_avg_pool2d(b, (1, 1)).flatten(1)
        b_max = F.adaptive_max_pool2d(b, (1, 1)).flatten(1)
        x4_avg = F.adaptive_avg_pool2d(x4, (1, 1)).flatten(1)
        x3_avg = F.adaptive_avg_pool2d(x3, (1, 1)).flatten(1)
        feat = torch.cat([b_avg, b_max, x4_avg, x3_avg], dim=1)
        return self.classifier(feat)

    def forward(self, x, return_attention=False, return_classification=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        x4, a4 = self.ag4(g=d4, x=e4)
        d4 = self.dec4(torch.cat((x4, d4), dim=1))

        d3 = self.up3(d4)
        x3, a3 = self.ag3(g=d3, x=e3)
        d3 = self.dec3(torch.cat((x3, d3), dim=1))

        d2 = self.up2(d3)
        x2, a2 = self.ag2(g=d2, x=e2)
        d2 = self.dec2(torch.cat((x2, d2), dim=1))

        d1 = self.up1(d2)
        x1, a1 = self.ag1(g=d1, x=e1)
        d1 = self.dec1(torch.cat((x1, d1), dim=1))

        out = self.final_conv(d1)

        cls_logits = self._compute_cls_logits(b, x4, x3) if return_classification else None

        if return_attention:
            a1_upsampled = F.interpolate(a1, size=x.shape[2:], mode='bilinear', align_corners=True)
            att_dict = {"ag1": a1_upsampled, "ag2": a2, "ag3": a3, "ag4": a4}
            if return_classification:
                return out, cls_logits, att_dict
            return out, att_dict

        if return_classification:
            return out, cls_logits
        return out


class UNet(nn.Module):
    """Baseline Multi-Task U-Net Architecture (Ronneberger et al., 2015) for HAM10000 Ablation"""
    def __init__(self, in_channels=3, out_channels=1, num_classes=7, features=[32, 64, 128, 256]):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        curr_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(features[-1] * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_classification=False):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        b = self.bottleneck(x)
        cls_logits = self.classifier(b) if return_classification else None

        x = b
        skips = skips[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = self.ups[idx + 1](torch.cat((skip, x), dim=1))

        out = self.final_conv(x)
        if return_classification:
            return out, cls_logits
        return out


class DiceBCELoss(nn.Module):
    """Hybrid Dice + Binary Cross-Entropy Loss for Medical Image Segmentation"""
    def __init__(self, dice_weight=0.6, bce_weight=0.4, smooth=1.0):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1.0 - dice_score.mean()
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class MultiTaskHAM10000Loss(nn.Module):
    """
    Joint Segmentation (Hybrid Dice + BCE) and 7-Class HAM10000 Diagnostic
    Classification (Class-Weighted Label-Smoothed Cross-Entropy) Loss.
    """
    def __init__(self, class_weights=None, seg_weight=1.0, cls_weight=0.5, label_smoothing=0.05):
        super(MultiTaskHAM10000Loss, self).__init__()
        self.seg_loss_fn = DiceBCELoss(dice_weight=0.6, bce_weight=0.4)
        self.cls_loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight

    def forward(self, seg_logits, cls_logits, mask_targets, label_targets):
        loss_seg = self.seg_loss_fn(seg_logits, mask_targets)
        loss_cls = self.cls_loss_fn(cls_logits, label_targets)
        total_loss = self.seg_weight * loss_seg + self.cls_weight * loss_cls
        return total_loss, loss_seg, loss_cls


def compute_segmentation_metrics(seg_logits, mask_targets, threshold=0.5, eps=1e-7):
    """Computes batch-averaged Dice (DSC), IoU (Jaccard), Precision, and Recall."""
    with torch.no_grad():
        preds = (torch.sigmoid(seg_logits) > threshold).float()
        targets = (mask_targets > 0.5).float()

        p_flat = preds.view(preds.size(0), -1)
        t_flat = targets.view(targets.size(0), -1)

        tp = (p_flat * t_flat).sum(dim=1)
        fp = (p_flat * (1.0 - t_flat)).sum(dim=1)
        fn = ((1.0 - p_flat) * t_flat).sum(dim=1)

        dice = ((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)).mean().item()
        iou = ((tp + eps) / (tp + fp + fn + eps)).mean().item()
        precision = ((tp + eps) / (tp + fp + eps)).mean().item()
        recall = ((tp + eps) / (tp + fn + eps)).mean().item()

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall
    }
