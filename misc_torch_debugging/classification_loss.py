import torch
import torch.nn as nn
import torch.nn.functional as F

# Simulate model output (logits for 3 classes)
model = nn.Linear(10, 3)
x = torch.randn(4, 10)
y = torch.tensor([1, 2, 0, 1])  # Ground truth class indices

logits = model(x)

# ❌ Bug: Using MSELoss for classification (wrong!)
# MSE expects float targets, not class indices
# and softmax output isn't ideal for regression losses
# --- BAD ---
# loss_fn = nn.MSELoss()
# loss = loss_fn(F.softmax(logits, dim=1), y)

# ✅ Fix: Use CrossEntropyLoss
# This combines LogSoftmax + NLLLoss internally
loss_fn = nn.MSELoss() 
try: 
    loss = loss_fn(F.softmax(logits, dim=1), y) # y is int, incompatible 
except Exception as e: 
    print("Error:", e)


loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, y)

print("Loss:", loss.item())
