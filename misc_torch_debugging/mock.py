import torch
import torch.nn as nn

x = torch.randn(32, 128)   # batch_size=32, features=128
w = torch.randn(128, 64)   # intended weight matrix

output = torch.matmul(x, w)
print(output.shape)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

"""
y = x
y += 2.0   # in-place operation  # ⚠️ in-place operation on a leaf tensor with requires_grad=True
    # The line y += 2.0 is an in-place operation that modifies x directly, since y just references x.
"""
y = x + 2.0   # creates a new tensor, out-of-place

z = y.sum()
z.backward()

# PyTorch autograd prohibits in-place modification of leaf tensors that require gradients, 
# because it needs the original value of x to compute gradients correctly during backpropagation.
print(x.grad)  # prints tensor([1., 1., 1.])

# Avoid in-place ops (like +=, *=, .add_()) on tensors with requires_grad=True — especially leaf tensors.
# Always prefer out-of-place ops (x + 2, x * y, etc.) unless you're sure in-place is safe (rare in gradient-tracked code).


"""
Case # 3 


"""
x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)

relu = nn.ReLU(inplace=False)
y = relu(x)
z = y.sum()
z.backward()
print("case #3", x.grad)

x = torch.randn(3, requires_grad=True)
y = x * 2
z = nn.ReLU(inplace=True)(y)
w = y + z  # uses y again   Then w = y + z tries to use the original y, but it’s been overwritten
# relu = nn.ReLU(inplace=False)
# or manually: z = nn.ReLU()(y.clone())  # safe, keeps y unchanged

loss = w.sum()
loss.backward()

print("case 3 loss: ", loss)

"""
1️⃣ Wrong Tensor Shape in Forward Pass
"""
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        # Bug: forgetting to flatten inputs for FC
        x = self.fc1(x)  # Expect shape [batch, 10]
        print(" steo shape: ", x.shape)
        return self.fc2(x)

model = SimpleNet()
# x = torch.randn(4, 2, 5)  # Incorrect shape (should be [batch, 10])
x= torch.randn(4, 10)
try:
    out = model(x)
except Exception as e:
    print("Error:", e)

# Fix: flatten input correctly
x_fixed = x.view(x.size(0), -1)
out = model(x_fixed)
print("Output shape:", out.shape)


"""
2️⃣ Missing .to(device) or Device Mismatch
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Linear(10, 2) # model = nn.Linear(10, 2).to(device)
x = torch.randn(4, 10).to(device)  # input on device

try:
    out = model(x)  # model not moved to device!
except Exception as e:
    print("Error:", e)

# Fix:
model = model.to(device)
out = model(x)
print("Output device:", out.device)


"""
3️⃣ Incorrect Loss Function Usage (e.g. MSELoss for classification)
"""
import torch.nn.functional as F

model = nn.Linear(10, 3)
x = torch.randn(4, 10)
y = torch.tensor([1, 2, 0, 1])  # Class labels for 3 classes

logits = model(x)

# Bug: Using MSELoss for classification (wrong)
loss_fn = nn.MSELoss()
try:
    loss = loss_fn(F.softmax(logits, dim=1), y)  # y is int, incompatible
except Exception as e:
    print("Error:", e)

# Fix: Use CrossEntropyLoss and raw logits
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, y)
print("Loss:", loss.item())

"""
4️⃣ Unstable Training: Exploding Gradients / NaNs
"""
model = nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=1000)  # Too high LR!

x = torch.randn(4, 10)
y = torch.tensor([0,1,0,1])

loss_fn = nn.CrossEntropyLoss()

for _ in range(5):
    optimizer.zero_grad()
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item()}")  # Watch for NaNs or Inf

# Fix: Reduce learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

"""
5️⃣ Forgetting model.train() or model.eval()
"""
model = nn.Sequential(
    nn.Linear(10, 10),
    nn.BatchNorm1d(10),
    nn.ReLU(),
    nn.Linear(10, 2)
)

x = torch.randn(4, 10)

# Bug: Forgetting model.eval() during validation, BatchNorm behaves incorrectly
model.train()  # should be train during training

out_train = model(x)

model.eval()   # Important for eval
out_eval = model(x)

print("Train output mean:", out_train.mean().item())
print("Eval output mean:", out_eval.mean().item())


"""
✅ Corrected Code with optimizer.zero_grad() in Loop
"""
model = nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

x = torch.randn(4, 10)
y = torch.tensor([0, 1, 0, 1])

for step in range(3):
    logits = model(x)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()            # ✅ Clear old gradients

    loss.backward()
    optimizer.step()
    # Bug: missing optimizer.zero_grad(), gradients accumulate!

    print(f"Step {step+1} loss:", loss.item())

# Fix:
optimizer.zero_grad()
for step in range(3):
    logits = model(x)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()  # Zero gradients before backward
    loss.backward()
    optimizer.step()
    print(f"Step {step+1} loss:", loss.item())

"""
✅ Example: Dropout in Training vs. Eval
"""
import torch
import torch.nn as nn

# Simple model with Dropout
class DropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.dropout = nn.Dropout(p=0.5)  # Drop 50% during training
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)  # Affected by train/eval mode
        x = self.fc2(x)
        return x

# Input tensor
x = torch.randn(1, 10)  # One sample

model = DropoutNet()

# 🔄 Run model multiple times in training mode
model.train()
print("Training mode outputs (dropout active):")
for _ in range(3):
    out = model(x)
    print(out)

# ✅ Switch to eval mode — dropout disabled
model.eval()
print("\nEvaluation mode outputs (dropout disabled):")
for _ in range(3):
    out = model(x)
    print(out)
