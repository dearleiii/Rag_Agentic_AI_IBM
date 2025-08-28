import torch
import torch.nn as nn
import torchvision.models as models

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# ✅ Fix 1: Convert grayscale input to 3-channel by repeating
x = torch.randn(8, 1, 224, 224)         # batch of 8 grayscale images
x_fixed = x.repeat(1, 3, 1, 1)          # convert to (8, 3, 224, 224)

# Test forward pass
try:
    out = model(x_fixed)
    print("✅ Forward pass successful. Output shape:", out.shape)
except Exception as e:
    print("❌ Forward pass failed:", e)

# ✅ Fix 2: Replace final FC layer properly
# Check current in_features before replacing
in_features = model.fc.in_features      # usually 512 for ResNet18
print("model.fc: ", model.fc)

model.fc = nn.Linear(in_features, 5)    # for 5 target classes
print("model.fc: ", model.fc)
# Test final output shape
out = model(x_fixed)
print("✅ Final output shape (after FC replacement):", out.shape)
