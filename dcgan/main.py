import torch
import torchvision
import torchvision.transforms as T
from dataclasses import dataclass

from .model import Generator, Discriminator


@dataclass
class Config:
    epochs = 10
    batch_size = 32
    lr = 2e-4

    img_size = (32, 32)

    z_dim = 100
    base_features = 128
    num_blocks = 3
    img_channels = 1


config = Config()

transform = T.Compose(
    [T.Resize(config.img_size), T.ToTensor(), T.Normalize([0.5], [0.5])]
)

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=config.batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=config.batch_size, shuffle=False
)

device = "cuda" if torch.cuda.is_available() else "cpu"

gen = Generator(
    z_dim=config.z_dim,
    g_channels=config.base_features,
    out_channels=config.img_channels,
    num_blocks=config.num_blocks,
).to(device)
disc = Discriminator(
    in_channels=config.img_channels,
    d_channels=config.base_features,
    num_blocks=config.num_blocks,
).to(device)

gen_optim = torch.optim.Adam(params=gen.parameters(), lr=config.lr)
disc_optim = torch.optim.Adam(params=disc.parameters(), lr=config.lr)
criterion = torch.nn.BCELoss()

gen_losses = []
disc_losses = []

for epoch in range(config.epochs):
    gen_running_loss = []
    disc_running_loss = []

    for batch_idx, (real, _) in enumerate(train_loader):
        real = real.to(device)

        # discriminator: max log D(real) + log(1 - D(G(z)))
        noise = torch.randn((config.batch_size, config.z_dim, 1, 1)).to(device)
        fake = gen(noise).detach()

        D_real = disc(real)
        logD_real = criterion(D_real, torch.ones_like(D_real))

        D_fake = disc(fake)
        logD_fake = criterion(D_fake, torch.zeros_like(D_fake))

        D_loss = (logD_real + logD_fake) / 2
        disc_running_loss.append(D_loss)

        disc_optim.zero_grad()
        D_loss.backward()
        disc_optim.step()

        # generator: min log(1 - D(G(z))) <-> max log D(G(z))
        noise = torch.randn((config.batch_size, config.z_dim, 1, 1)).to(device)
        fake = gen(noise)

        D_fake = disc(fake)
        G_loss = criterion(D_fake, torch.ones_like(D_fake))
        gen_running_loss.append(G_loss)

        gen_optim.zero_grad()
        G_loss.backward()
        gen_optim.step()

        if (
            len(gen_running_loss) == 200 and len(disc_running_loss) == 200
        ) or batch_idx == len(train_loader) - 1:
            gen_avg_loss = sum(gen_running_loss) / len(gen_running_loss)
            disc_avg_loss = sum(disc_running_loss) / len(disc_running_loss)

            gen_losses.append(gen_avg_loss)
            gen_running_loss = []

            disc_losses.append(disc_avg_loss)
            disc_running_loss = []

            print(
                f"[epoch {epoch + 1}, batch idx {batch_idx + 1}] gen loss = {gen_avg_loss}, disc loss = {disc_avg_loss}"
            )
