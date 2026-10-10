"""
Model components for ADNI AD vs NC classification.

BaselineCNN is a deliberately simple reference model. The advanced model
(ConvNeXt) will be added to this file later and compared against it on the
same patient-level splits.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv(3x3) -> BatchNorm -> ReLU -> MaxPool(2). Halves the spatial size."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    """Four-stage plain CNN baseline.

    Input : (B, 1, 224, 224) grayscale MRI slice
    Output: (B, num_classes) logits (NC=0, AD=1)
    """

    def __init__(self, in_ch=1, num_classes=2, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        layers, prev = [], in_ch
        for w in widths:                      # 224 -> 112 -> 56 -> 28 -> 14
            layers.append(ConvBlock(prev, w))
            prev = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)   # global average pooling -> (B, C, 1, 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(prev, num_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_model(name, num_classes=2):
    """Factory so train.py / predict.py can select a model by name."""
    if name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
