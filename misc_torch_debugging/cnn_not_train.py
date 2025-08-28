import torch
import torch.nn as nn
import torch.optim as optim

class BadCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3) # kernel_size = 3 → padding = 0 (default), stride = 1/ in_channels, out_channels, kernel_size
        self.pool = nn.MaxPool2d(2) # kernel_size = 2 → downsample by 2x
        # self.fc = nn.Linear(64, 10)     # self.fc = nn.Linear(16 * 15 * 15, 10)   # 3600 → 10
        self.fc = nn.Linear(16 * 15 * 15, 10)
        
    def forward(self, x):
        # x = torch.randn(4, 3, 32, 32)
        x = self.conv(x)
        print("x shape: ", x.shape) #torch.Size([4, 16, 30, 30])
        x = self.pool(x) 
        print("x shape: ", x.shape) #torch.Size([4, 16, 15, 15])

        x = x.view(x.size(0), -1)  # flatten torch.Size([4, 3600])
        print("x shape: ", x.shape)
        return self.fc(x)

model = BadCNN()
x = torch.randn(4, 3, 32, 32)
out = model(x)
print(out.shape)
