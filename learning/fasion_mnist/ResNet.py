import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

transform = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

training_dataset = datasets.ImageNet(
    root="./data",
    split="train",
    transform=transform,
)

train_dataloader = DataLoader(training_dataset, batch_size=64)

device = "cuda"
model = models.resnet50(pretrained=True)
for parma in model.parameters():
    parma.requires_grad = False
model.fc = torch.nn.Linear(2048, 2)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

epoch = 5


for i in range(epoch):
    pass
