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
    """Attention U-Net Architecture (Oktay et al., 2018)"""
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256], dropout_rate=0.1):
        super(AttentionUNet, self).__init__()
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

    def forward(self, x, return_attention=False):
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

        if return_attention:
            a1_upsampled = F.interpolate(a1, size=x.shape[2:], mode='bilinear', align_corners=True)
            return out, {"ag1": a1_upsampled, "ag2": a2, "ag3": a3, "ag4": a4}
        return out

class UNet(nn.Module):
    """Baseline U-Net Architecture (Ronneberger et al., 2015)"""
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super(UNet, self).__init__()
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

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = self.ups[idx + 1](torch.cat((skip, x), dim=1))

        return self.final_conv(x)
