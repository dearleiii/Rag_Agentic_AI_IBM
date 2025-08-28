"""
Let’s define a small neural network where we accidentally use in-place ReLU, 
and later reuse the same tensor, causing an autograd failure.
"""

import torch
import torch.nn as nn

# Define a simple model
class BuggyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.relu = nn.ReLU(inplace=False)  # <- in-place activation     self.relu = nn.ReLU(inplace=False)  # <-- changed this
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        out = self.linear1(x)
        activated = self.relu(out.clone())      # modifies `out` in-place
        combined = out + activated      # `out` is reused after in-place op
        return self.linear2(combined)

# Dummy input
x = torch.randn(5, 10, requires_grad=True)
model = BuggyNet()

# Forward and backward pass
output = model(x)
loss = output.mean()
loss.backward()
print("output dim: ", output.shape)
print("loss : ", loss, loss.grad)