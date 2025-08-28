"""
Design a CNN model to classify 10 categories from 64x64 RGB images 
on an edge device. Memory must be < 10MB. Training time is not a concern.

Goal:

- Pick efficient ops (depthwise convs, fewer params)
- Decide on architecture: number of layers, activations, BN?
- Justify: Why not use ResNet18?

🧠 Step 1: Why NOT ResNet18?

ResNet18:
~11.7M parameters → ~45MB (float32)
Uses many full conv layers and residual blocks — not optimized for edge

Instead, we’ll design a tiny CNN with:

- Depthwise separable convolutions
- Few full conv layers
- BatchNorm + ReLU
- Possibly global average pooling to reduce FC params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyEdgeCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def dw_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=False),  # Depthwise
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),  # Pointwise
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # [B, 16, 32, 32]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            dw_conv(16, 32),       # [B, 32, 32, 32]
            nn.MaxPool2d(2),       # [B, 32, 16, 16]

            dw_conv(32, 64),       # [B, 64, 16, 16]
            nn.MaxPool2d(2),       # [B, 64, 8, 8]

            dw_conv(64, 128),      # [B, 128, 8, 8]
            nn.AdaptiveAvgPool2d(1)  # [B, 128, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
model = TinyEdgeCNN()
num_params = sum(p.numel() for p in model.parameters())
size_MB = num_params * 4 / 1e6  # float32
print(f"Total params: {num_params:,} (~{size_MB:.2f} MB)")

