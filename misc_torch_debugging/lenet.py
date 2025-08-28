import torch
import torch.nn as nn
import torch.nn.functional as F

class BuggyLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Bug: Conv layer has wrong output channels for MNIST 10 classes task
        self.conv1 = nn.Conv2d(1, 20, 5)  # okay
        self.conv2 = nn.Conv2d(20, 10, 5) # Bug: too few output channels

        self.fc1 = nn.Linear(10 * 4 * 4, 50)  # Bug: may not match flattened size
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        print("x shape: ", x.shape) # x shape:  torch.Size([8, 20, 24, 24])
        x = F.max_pool2d(x, 2)
        print("x shape: ", x.shape) # x shape:  torch.Size([8, 20, 12, 12])
        x = self.conv2(x)          # Bug: missing activation here
        print("x shape: ", x.shape) 
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        print("x shape: ", x.shape) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)            # Bug: missing softmax if used with wrong loss
        return x

# Debugging steps:
# - Fix conv2 out_channels
# - Fix fc1 input size
# - Add activation after conv2
# - Use CrossEntropyLoss (which has softmax built-in)
model = BuggyLeNet()
criterion = nn.CrossEntropyLoss()  # accepts raw logits

x = torch.randn(8, 1, 28, 28)  # batch of 8 MNIST images
y = torch.randint(0, 10, (8,))  # labels

logits = model(x)
loss = criterion(logits, y)
loss.backward()
print("Loss:", loss.item())
