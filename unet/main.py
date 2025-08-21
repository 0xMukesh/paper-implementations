import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import os

from unet.datatset import TeethSegmentationDataset
from unet.model import UNet
from unet.utils import calculate_accuracy

DATASET_ROOT = "/home/mukesh/.cache/kagglehub/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images/versions/1/Teeth Segmentation PNG/d2"
IMGS_DIR = "img"
MASKS_DIR = "masks_machine"
CHECKPOINTS_DIR = "checkpoints"
TARGET_SIZE = (512, 256)
BATCH_SIZE = 32
NUM_EPOCHS = 70
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
PIN_MEMORY = True
NUM_WORKERS = 2

train_transform = T.Compose(
    [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ToTensor()]
)
test_transform = T.Compose([T.ToTensor()])

train_dataset = TeethSegmentationDataset(
    root=DATASET_ROOT,
    imgs_dir=IMGS_DIR,
    masks_dir=MASKS_DIR,
    target_size=TARGET_SIZE,
    split="train",
)
test_dataset = TeethSegmentationDataset(
    root=DATASET_ROOT,
    imgs_dir=IMGS_DIR,
    masks_dir=MASKS_DIR,
    target_size=TARGET_SIZE,
    split="val",
)

train_data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=PIN_MEMORY,
    num_workers=NUM_WORKERS,
)
test_data_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=PIN_MEMORY,
    num_workers=NUM_WORKERS,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(in_channels=1, num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, patience=PATIENCE
)

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

best_dice_score = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_losses = []

    loop = tqdm(enumerate(train_data_loader), total=len(train_data_loader), leave=True)

    for i, (img, mask) in loop:
        img = img.to(device)
        mask = mask.to(device)

        predication = model(img).to(device)
        loss = criterion(predication, mask)

        running_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"epoch [{epoch + 1}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())

    acc, dice_score = calculate_accuracy(model, test_data_loader, device)
    mean_loss = sum(running_losses) / len(running_losses)

    scheduler.step(mean_loss)

    if dice_score > best_dice_score:
        best_dice_score = dice_score
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": mean_loss,
            "dice_score": dice_score,
        }

        best_model_path = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
        torch.save(checkpoint, best_model_path)

        print(f"best dice score updated ({best_dice_score}). checkpoint saved")

    print(f"\nsummary for {epoch + 1} epoch")
    print(f"  mean loss: {mean_loss}")
    print(f"  acc: {acc:.4f}")
    print(f"  dice score: {dice_score:.4f}")
