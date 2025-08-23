import torch
from torch import nn
from torch.nn import functional as F

from .utils import calculate_dice_score


class BCEDiceLoss(nn.Module):
    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()

        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        bce = F.binary_cross_entropy_with_logits(pred, target)
        dice_loss = 1 - calculate_dice_score(pred, target)

        return (1 - self.alpha) * bce + self.alpha * dice_loss
