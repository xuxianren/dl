import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import time

plt.rcParams["font.sans-serif"] = ["kaiti"]

seed = 999
torch.manual_seed(seed)

dataroot = "data/celeba"
workders = 8
batch_size = 256
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 5
lr = 0.0002
ngpu = 1
beta1 = 0.5 * ngpu

transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ),
    ]
)

dataset = ImageFolder(
    root=dataroot,
    transform=transform,
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workders,
)

device = "cuda"


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入是Z
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # size = (ngf*8, 4, 4)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # size = (ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # size = (ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # size = (ngf, 32, 32)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # size = (nc, 64, 64)
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # size = (nc, 64, 64)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf, 32, 32)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2, 16, 16)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4, 8, 8)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8, 4, 4)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
# print(netG)

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
# print(netD)

loss_fn = nn.BCELoss()

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []


def train():
    print("开始训练")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            # 更新D, 最大化 log(D(x)) + log(1-D(G(z))
            # 用真数据训练D
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full(
                (b_size,),
                real_label,
                dtype=torch.float32,
                device=device,
            )
            #
            # print(real_cpu.shape, label.shape)
            output = netD(real_cpu).view(-1)
            errD_real = loss_fn(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # 用假数据训练D
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = loss_fn(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # 更新G  最大化 log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = loss_fn(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print(
                    "%s [%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    % (
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        epoch,
                        num_epochs,
                        i,
                        len(dataloader),
                        errD.item(),
                        errG.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    )
                )
            # Save Losses for plotting later
            # G_losses.append(errG.item())
            # D_losses.append(errD.item())
            # Check how the generator is doing by saving G's output on fixed_noise
            if i % 500 == 0:
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    plt.imsave(
                        f"./data/fake/{epoch}_{i}.png",
                        np.transpose(
                            vutils.make_grid(fake, padding=2, normalize=True).cpu(),
                            (1, 2, 0),
                        ).numpy(),
                    )


if __name__ == "__main__":
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.title("训练图像")
    # plt.imshow(
    #     np.transpose(
    #         vutils.make_grid(
    #             real_batch[0].to(device)[:64], padding=2, normalize=True
    #         ).cpu(),
    #         (1, 2, 0),
    #     )
    # )
    # plt.show()
    # plt.imsave(
    #     "real_1.png",
    #     np.transpose(
    #         vutils.make_grid(
    #             real_batch[0].to(device)[:64], padding=2, normalize=True
    #         ).cpu(),
    #         (1, 2, 0),
    #     ).numpy(),
    # )
    train()
    # plt.figure(figsize=(10, 5))
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(G_losses, label="G")
    # plt.plot(D_losses, label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()
