import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchmetrics.segmentation import DiceScore
import matplotlib.pyplot as plt
import random
from typing import cast, Literal, List


class CombinedTransform:
    def __init__(
        self,
        rotation_degrees: float = 0.0,
        h_flip_prob: float = 0.5,
        v_flip_prob: float = 0.5,
        img_additional_transform=None,
        mask_additional_transform=None,
    ) -> None:
        self.rotation_degrees = rotation_degrees
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.img_additional_transform = img_additional_transform
        self.mask_additional_transform = mask_additional_transform

    def __call__(self, img, mask):
        img, mask = self._apply_synchronized_transforms(img, mask)

        if self.img_additional_transform:
            img = self.img_additional_transform(img)

        if self.mask_additional_transform:
            mask = self.mask_additional_transform(mask)

        return img, mask

    def _apply_synchronized_transforms(self, img, mask):
        if random.random() < self.h_flip_prob:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        if random.random() < self.v_flip_prob:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        if self.rotation_degrees > 0:
            degrees = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            img = TF.rotate(img, degrees, fill=[0.0])
            mask = TF.rotate(mask, degrees, fill=[0.0])

        return img, mask


def run_inference(model: nn.Module, loader: DataLoader, device: Literal["cuda", "cpu"]):
    model.eval()
    dice = DiceScore(num_classes=1, average="micro")
    dice_scores = []

    with torch.no_grad():
        for img, mask in loader:
            img = cast(torch.Tensor, img.to(device))
            mask = cast(torch.Tensor, mask.to(device))

            pred = torch.sigmoid(model(img))
            dice_score = dice(pred, mask)

            dice_scores.append(dice_score.item())

    return torch.median(torch.tensor(dice_scores)).item()


def plot_loss_curve(batch_losses: List[float], epoch_avg_losses: List[float]):
    _, axes = plt.subplots(1, 2, figsize=(15, 10))

    axes[0].plot(batch_losses)
    axes[0].set_xlabel("loss")
    axes[0].set_ylabel("batch")

    axes[1].plot(epoch_avg_losses)
    axes[1].set_xlabel("loss")
    axes[1].set_ylabel("epoch")

    plt.tight_layout()
    plt.show()
