import torch
import torch.nn as nn
import torch.nn.functional as F  # Needed for relu

x = torch.randn(16, 1, 28, 28)
y = torch.randint(0, 10, (16,))

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, 5) 
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(10 * 12 * 12, 10)  # 28-5+1=24 -> 24/2=12

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        print("x shape: ", x.shape)     # x shape:  torch.Size([16, 10, 12, 12])
        x = x.view(-1, 10 * 12 * 12)  # flatten
        print("x shape: ", x.shape)  # x shape:  torch.Size([16, 1440])

        x = self.fc(x)
        return x

model = SimpleCNN()
logits = model(x)

# Buggy usage example:
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, y.squeeze())  # Bug: y is already correct shape, no squeeze needed
# loss = loss_fn(logits, y)

# Fix: remove squeeze
loss = loss_fn(logits, y)
loss.backward()  # Backpropagate to test it's working

print("Loss:", loss.item())
