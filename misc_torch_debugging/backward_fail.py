import torch
import torch.nn as nn

x = torch.randn(16, 100)
y = torch.randint(0, 2, (16,))

model = nn.Sequential(
    nn.Linear(100, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    # nn.Sigmoid()
)

loss_fn = nn.BCELoss()  # loss_fn = nn.BCEWithLogitsLoss()  # combines Sigmoid + BCELoss in one step

optimizer = torch.optim.Adam(model.parameters())    

y_pred = model(x).squeeze()  # y_pred = model(x).view(-1)  # always ensures shape [batch_size]

print("output shape: ", y_pred.shape)   # output shape:  torch.Size([16])
loss = loss_fn(y_pred, y.float())
# print("loss: ", loss, loss.backward())  #loss:  tensor(0.7257, grad_fn=<BinaryCrossEntropyBackward0>) None
loss.backward()
optimizer.step()
print("loss: ", loss)

"""
You're using nn.BCELoss() (Binary Cross Entropy) with Sigmoid in the model, 
which is okay if the shapes match — 
but this code can silently fail or behave incorrectly due to shape mismatch.
"""