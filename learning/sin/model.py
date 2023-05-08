import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


class Model1(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main = nn.Sequential(
            nn.Linear(1, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            # nn.ReLU(),
            # nn.Linear(1, 1),
            # nn.ReLU(),
            # nn.Linear(1, 1),
            # nn.ReLU(),
            # nn.Linear(1, 1),
            #nn.ReLU(),
            #nn.Linear(5, 1),
            # nn.ReLU(),
            # nn.Linear(5, 1),
            # nn.ReLU(),
            # nn.Linear(1, 1),
        )

    def forward(self, x):
        return self.main(x)


model = Model1()
model.to("cuda")

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def gen_data():
    x = torch.arange(0, 4*torch.pi, 0.1)
    x = x.reshape((-1, 1))
    y = torch.sin(x)
    e = torch.normal(mean=0, std=0.05, size=(len(y), 1))
    # plt.plot(x, y+e)
    # plt.show()
    return x, y+e


x, y = gen_data()
print(x.shape, y.shape)
training_dataset = TensorDataset(x, y)

training_dataload = DataLoader(
    training_dataset, batch_size=len(training_dataset))

epoch_nums = 10000


def train():
    for epoch in range(epoch_nums, -1, -1):
        for X_c, y_c in training_dataload:
            X, y = X_c.to("cuda"), y_c.to("cuda")
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch == 0:
                with torch.no_grad():
                    y_pred = model(X).to("cpu")
                    plt.plot(X_c, y_pred)


train()
plt.plot(x, y)
plt.show()
