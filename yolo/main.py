import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from yolo.dataset import VOCDataset
from yolo.loss import YOLOLoss
from yolo.model import ARCH_CONFIG, YOLOv1

EPOCHS = 1000
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = 2
PIN_MEMORY = True

train_transform = T.Compose([T.Resize((448, 448)), T.ToTensor()])
test_transform = T.Compose([T.Resize((448, 448)), T.ToTensor()])

train_dataset = VOCDataset(
    root="voc", csv_file="8examples.csv", transform=train_transform, header=0
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

test_dataset = VOCDataset(root="voc", csv_file="test.csv", transform=test_transform)
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
    mean_losses = {"box_loss": [], "obj_loss": [], "no_obj_loss": [], "class_loss": []}

    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)

    for i, (img, targets) in loop:
        img = img.to(DEVICE)
        targets = targets.to(DEVICE)

        predictions = model(img)

        loss, box_loss, obj_loss, no_obj_loss, class_loss = criterion(
            predictions, targets
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        mean_losses["box_loss"].append(box_loss.item())
        mean_losses["obj_loss"].append(obj_loss.item())
        mean_losses["no_obj_loss"].append(no_obj_loss.item())
        mean_losses["class_loss"].append(class_loss.item())

        loop.set_description(f"epoch [{epoch+1}/{EPOCHS}]")
        loop.set_postfix(
            loss=loss.item(),
            box_loss=box_loss.item(),
            obj_loss=obj_loss.item(),
            no_obj_loss=no_obj_loss.item(),
            class_loss=class_loss.item(),
        )

    print(f"\nmean losses for epoch {epoch + 1}:")
    print(
        f"  box loss: {sum(mean_losses["box_loss"])/len(mean_losses["box_loss"]):.4f}"
    )
    print(
        f"  object loss: {sum(mean_losses["obj_loss"])/len(mean_losses["obj_loss"]):.4f}"
    )
    print(
        f"  no object loss: {sum(mean_losses["no_obj_loss"])/len(mean_losses["no_obj_loss"]):.4f}"
    )
    print(
        f"  class loss: {sum(mean_losses["class_loss"])/len(mean_losses["class_loss"]):.4f}"
    )

    
