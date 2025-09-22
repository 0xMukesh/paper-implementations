import torch
import torchvision
import torchvision.transforms as T
from dataclasses import dataclass

from .model import Generator, Critic
from .utils import calculate_grad_penalty


@dataclass
class Config:
    epochs = 5
    batch_size = 64
    lr_gen = 1e-4
    lr_critic = 1e-4
    critic_iter = 5
    lambda_gp = 10

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

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=config.batch_size, shuffle=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"

gen = Generator(
    z_dim=config.z_dim,
    g_channels=config.base_features,
    out_channels=config.img_channels,
    num_blocks=config.num_blocks,
).to(device)

critic = Critic(
    in_channels=config.img_channels,
    d_channels=config.base_features,
    num_blocks=config.num_blocks,
).to(device)

optim_gen = torch.optim.Adam(
    params=gen.parameters(), lr=config.lr_gen, betas=(0.5, 0.9)
)
optim_critic = torch.optim.Adam(
    params=critic.parameters(), lr=config.lr_critic, betas=(0.5, 0.9)
)

gen_losses = []
critic_losses = []

for epoch in range(config.epochs):
    for batch_idx, (real, _) in enumerate(train_loader):
        real = real.to(device)
        batch_size = real.size(0)

        critic_loss_batch = 0

        for _ in range(config.critic_iter):
            noise = torch.randn((batch_size, config.z_dim, 1, 1)).to(device)
            fake = gen(noise).detach()

            C_real = critic(real)
            C_fake = critic(fake)

            gp = calculate_grad_penalty(critic, real, fake, device)

            C_loss = torch.mean(C_fake) - torch.mean(C_real) + config.lambda_gp * gp
            critic_loss_batch += C_loss.item()

            optim_critic.zero_grad()
            C_loss.backward()
            optim_critic.step()

        noise = torch.randn((batch_size, config.z_dim, 1, 1)).to(device)
        fake = gen(noise)

        G_loss = -torch.mean(critic(fake))

        optim_gen.zero_grad()
        G_loss.backward()
        optim_gen.step()

        gen_losses.append(G_loss.item())
        critic_losses.append(critic_loss_batch / config.critic_iter)

        if batch_idx % 100 == 0:
            print(
                f"[epoch {epoch+1}/{config.epochs}, batch {batch_idx}] "
                f"gen loss: {G_loss.item():.4f}, critic loss: {critic_loss_batch/config.critic_iter:.4f}"
            )

    print(f"epoch {epoch+1} completed")
