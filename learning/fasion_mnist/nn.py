from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    import torch
    from .data import train_dataloader, test_dataloader, classes
    from .train import train, test

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    X, y = test_dataloader.dataset[0]
    X = X.to(device)
    pred = model(X)
    print(f"预测结果为{classes[pred[0].argmax(0)]}, 实际结果为{classes[y]}")
