# train_data
# model
# train
#

from torch import nn


class LeNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        y = self.fc(x)
        return y


if __name__ == "__main__":
    import torch
    from .data import train_dataloader, test_dataloader, classes
    from .train import train, test

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LeNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)

    # def save_model(path):
    #     torch.save(model.state_dict(), path)
    # path = "./checkpoints/lenet.pth"
    # # save_model()
    # model.load_state_dict(torch.load(path))
    # model.eval()

    X, y = test_dataloader.dataset[0]
    X = X.to(device)
    X = X.view(1, 1, 28, 28)
    pred = model(X)
    print(f"预测结果为{classes[pred[0].argmax(0)]}, 实际结果为{classes[y]}")
