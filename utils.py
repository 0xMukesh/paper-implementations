import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

losses = []

def train_model_cifar10(model_fn, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_mean = [0.4914, 0.4822, 0.4465]
    dataset_std = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    train_dataset = datasets.CIFAR10(root="data", train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(root="data", train=False, transform=test_transform, download=True) 

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = model_fn().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            if isinstance(output, list):
                  loss = 0.1 * loss_function(output[0], labels) + 0.1 * loss_function(output[1], labels) + loss_function(output[2], labels)
            else:
                loss = loss_function(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                print(f"[epoch {epoch + 1}, batch {i + 1}] running loss - {running_loss / 100:.3f}")
                losses.append(running_loss)
                running_loss = 0

        with torch.no_grad():
          model.eval()
          correct = 0
          total = 0

          for i, (images, labels) in enumerate(test_loader):
              images = images.to(device)
              labels = labels.to(device)

              output = model(images)
              if isinstance(output, list):
                output = output[-1]

              _, predicted = torch.max(output, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()

          acc = correct / total
          print(f"[epoch {epoch + 1}] validation acc - {acc * 100:.3f}%")
