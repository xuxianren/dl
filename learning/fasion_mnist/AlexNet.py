from torch import nn
from torchvision import datasets

training_data = datasets.ImageNet(
    root="./data",
    split="train",
    download=True,
)


class AlexNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv3d(1, 96, (3, 11, 11), stride=4, padding=2),
            nn.MaxPool2d((3, 3), 2),
            nn.Conv3d(1, 256, (96, 5, 5), padding=2),
            nn.MaxPool2d((3, 3), 2),
            nn.Conv3d(1, 384, (256, 3, 3), padding=1),
            nn.Conv3d(1, 256, (384, 3, 3), padding=1),
            nn.MaxPool2d(3, 2),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.conv(x)
        return y
