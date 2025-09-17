import torch
from torch import nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from einops import rearrange
from dataclasses import dataclass


class Discriminator(nn.Module):
    def __init__(self, img_dim: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, latent_dim, img_dim) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class Config:
    epochs = 20
    batch_size = 64
    lr = 3e-4

    img_size = 28 * 28
    latent_size = 64


config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

train_dataset = torchvision.datasets.MNIST(
    root="./data/mnist", train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="./data/mnist", train=False, transform=transform, download=False
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=config.batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=config.batch_size, shuffle=False
)

disc = Discriminator(config.img_size).to(device)
gen = Generator(config.latent_size, config.img_size).to(device)
optim_disc = torch.optim.Adam(params=disc.parameters(), lr=config.lr)
optim_gen = torch.optim.Adam(params=gen.parameters(), lr=config.lr)
criterion = nn.BCELoss()

D_losses = []
G_losses = []

for epoch in range(config.epochs):
    for batch_idx, (real, _) in enumerate(train_loader):
        real = rearrange(real, "b c h w -> b (c h w)").to(device)

        # discriminator: max log D(real) + log(1 - D(G(z)))
        noise = torch.rand((config.batch_size, config.latent_size)).to(device)
        fake = gen(noise)

        D_real = disc(real).view(-1)
        logD_real = criterion(D_real, torch.ones_like(D_real))

        D_fake = disc(fake).view(-1)
        logD_fake = criterion(D_fake, torch.zeros_like(D_fake))

        D_loss = (logD_real + logD_fake) / 2

        optim_disc.zero_grad()
        D_loss.backward()
        optim_disc.step()

        # generator: min log(1 - D(G(z))) <-> max log D(G(z))
        noise = torch.rand((config.batch_size, config.latent_size)).to(device)
        fake = gen(noise)
        D_fake = disc(fake).view(-1)
        G_loss = criterion(D_fake, torch.ones_like(D_fake))

        optim_gen.zero_grad()
        G_loss.backward()
        optim_gen.step()

        if batch_idx % 100 == 0:
            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())

            print(
                f"epoch: {epoch}, batch idx: {batch_idx}: loss D: {D_loss.item()}, loss G: {G_loss.item()}"
            )

with torch.no_grad():
    noise = torch.rand((config.batch_size, config.latent_size)).to(device)
    fake = gen(noise).reshape(-1, 1, 28, 28)
    plt.imshow(fake[0].cpu().numpy().squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()
