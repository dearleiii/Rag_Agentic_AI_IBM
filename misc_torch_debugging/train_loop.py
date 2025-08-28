def train(model, dataloader, optimizer, loss_fn):
    for epoch in range(5):
        for batch in dataloader:
            x, y = batch
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
