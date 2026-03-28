import torch
import torch.nn as nn
import timm


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


class FusionModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.cnn = timm.create_model(
            "mobilenetv3_small_100", pretrained=True, num_classes=0
        )

        self.transformer = timm.create_model(
            "vit_tiny_patch16_224", pretrained=True, num_classes=0
        )

        # 🔴 AUTO DETECT DIMENSIONS (NO HARDCODING)
        cnn_dim = 1024
        trans_dim = 192
        total_dim = cnn_dim + trans_dim

        self.se = SEBlock(total_dim)

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)
        trans_feat = self.transformer(x)

        fused = torch.cat([cnn_feat, trans_feat], dim=1)
        fused = self.se(fused)

        return self.classifier(fused)