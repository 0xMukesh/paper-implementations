import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
import os
from typing import cast

from unet.model import UNet
from unet.dataset import LucchiDataset
from unet.loss import BCEDiceLoss
from unet.utils import CombinedTransform, plot_loss_curve, run_inference


np.random.seed(42)

DATASET_ROOT = "/content/Lucchi++"
TARGET_SIZE = (256, 192)
PIN_MEMORY = True
NUM_WORKERS = 2

NUM_EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4

CHECKPOINTS_DIR = "checkpoints"

additional_transform = T.Compose([T.Resize(TARGET_SIZE), T.ToTensor()])

train_transform = CombinedTransform(
    rotation_degrees=15.0,
    img_additional_transform=additional_transform,
    mask_additional_transform=additional_transform,
)
test_transform = CombinedTransform(
    rotation_degrees=0,
    h_flip_prob=0,
    v_flip_prob=0,
    img_additional_transform=additional_transform,
    mask_additional_transform=additional_transform,
)

train_dataset = LucchiDataset(
    root=DATASET_ROOT, split="train", combined_transform=train_transform
)
test_dataset = LucchiDataset(
    root=DATASET_ROOT, split="test", combined_transform=test_transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=PIN_MEMORY,
    num_workers=NUM_WORKERS,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=PIN_MEMORY,
    num_workers=NUM_WORKERS,
)

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(in_channels=1, num_classes=1).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)
loss_fn = BCEDiceLoss()

batch_losses = []
epoch_avg_losses = []

best_dice_score = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)

    for _, (img, mask) in loop:
        img = cast(torch.Tensor, img.to(device))
        mask = cast(torch.Tensor, mask.to(device))

        pred = model(img)
        loss = loss_fn(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_losses.append(loss.item())

        loop.set_description(f"epoch [{epoch+1}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    dice_score = run_inference(model, test_loader, device)
    scheduler.step(dice_score)

    epoch_avg_losses.append(running_loss / len(train_loader))

    if dice_score > best_dice_score:
        best_dice_score = dice_score
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_loss": avg_loss,
            "dice_score": dice_score,
        }

        torch.save(checkpoint, os.path.join(CHECKPOINTS_DIR, "best_dice_score.pth"))

    print(f"summary for {epoch} epoch")
    print(f"avg loss = {avg_loss:.4f}")
    print(f"dice score = {dice_score:.4f}")

plot_loss_curve(batch_losses, epoch_avg_losses)
