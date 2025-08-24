import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.segmentation import DiceScore


class BCEDiceLoss(nn.Module):
    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target)
        probs = torch.sigmoid(pred)
        dice = DiceScore(num_classes=1, average="micro")
        dice_loss = 1 - dice(probs, target)

        return (1 - self.alpha) * bce + self.alpha * dice_loss
