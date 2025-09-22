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
    lr = 1e-4
    weight_clamp = 0.01
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
critic = Critic(
    in_channels=config.img_channels,
    d_channels=config.base_features,
    num_blocks=config.num_blocks,
).to(device)

optim_gen = torch.optim.Adam(params=gen.parameters(), lr=config.lr, betas=(0.0, 0.9))
optim_critic = torch.optim.Adam(
    params=critic.parameters(), lr=config.lr, betas=(0.0, 0.9)
)

gen_losses = []
critic_losses = []

for epoch in range(config.epochs):
    gen_running_loss = []
    critic_running_loss = []

    for batch_idx, (real, _) in enumerate(train_loader):
        real = real.to(device)

        # discriminator: max E[critic(real)] - E[critic(fake)]
        for _ in range(config.critic_iter):
            noise = torch.randn((real.size(0), config.z_dim, 1, 1)).to(device)
            fake = gen(noise)

            C_real = critic(real)
            C_fake = critic(fake.detach())

            gp = calculate_grad_penalty(critic, real, fake, device)

            C_loss = torch.mean(C_fake) - torch.mean(C_real) + config.lambda_gp * gp
            critic_running_loss.append(C_loss.item())

            optim_critic.zero_grad()
            C_loss.backward()
            optim_critic.step()

            for p in critic.parameters():
                p.data.clamp_(-config.weight_clamp, config.weight_clamp)

        # generator: max E[critic(gen_fake)] <-> -min E[critic(gen_fake)]
        noise = torch.randn((real.size(0), config.z_dim, 1, 1)).to(device)
        fake = gen(noise)

        G_loss = -torch.mean(critic(fake))
        gen_running_loss.append(G_loss.item())

        optim_gen.zero_grad()
        G_loss.backward()
        optim_gen.step()

        if (
            len(gen_running_loss) == 200 and len(critic_running_loss) == 200
        ) or batch_idx == len(train_loader) - 1:
            gen_avg_loss = sum(gen_running_loss) / len(gen_running_loss)
            critic_avg_loss = sum(critic_running_loss) / len(critic_running_loss)

            gen_losses.append(gen_avg_loss)
            gen_running_loss = []

            critic_losses.append(critic_avg_loss)
            critic_running_loss = []

            print(
                f"[epoch {epoch + 1}, batch idx {batch_idx + 1}] gen loss = {gen_avg_loss}, critic loss = {critic_avg_loss}"
            )
