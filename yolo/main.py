import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from yolo.dataset import VOCDataset
from yolo.loss import YOLOLoss
from yolo.model import ARCH_CONFIG, YOLOv1

EPOCHS = 200
LEARNING_RATE = 2e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True

transform = T.Compose([T.Resize((448, 448)), T.ToTensor()])

train_dataset = VOCDataset(root="voc", csv_file="train.csv", transform=transform)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

test_dataset = VOCDataset(root="voc", csv_file="test.csv", transform=transform)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

model = YOLOv1(ARCH_CONFIG).to(DEVICE)
criterion = YOLOLoss().to(DEVICE)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    loop = tqdm(enumerate(train_loader), total=len(train_loader))

    for i, (img, targets) in loop:
        img = img.to(DEVICE)
        targets = targets.to(DEVICE)

        predictions = model(img)
        loss = criterion(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()

        loop.set_description(
            f"epoch: {epoch + 1}/{EPOCHS} | batch: {i + 1}/{len(train_loader)}"
        )
        loop.set_postfix(loss=loss.item())
