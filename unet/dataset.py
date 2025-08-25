import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from typing import Literal

from unet.utils import CombinedTransform


class CarvanaDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "test"],
        combined_transform: CombinedTransform | None = None,
    ) -> None:
        super().__init__()

        self.root = root
        self.combined_transform = combined_transform

        self.img_dir = os.path.join(self.root, "train_images")
        self.mask_dir = os.path.join(self.root, "train_masks")

        basenames = [
            f.split(".")[0] for f in os.listdir(self.img_dir) if f.endswith(".jpg")
        ]
        basenames = sorted(basenames)

        np.random.shuffle(basenames)

        split_idx = int(len(basenames) * 0.8)
        basenames = basenames[:split_idx] if split == "train" else basenames[split_idx:]

        self.img_files = [os.path.join(self.img_dir, f + ".jpg") for f in basenames]
        self.mask_files = [os.path.join(self.mask_dir, f + ".png") for f in basenames]

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert("RGB")
        mask = Image.open(self.mask_files[idx])
        mask = Image.fromarray(np.asarray(mask) * 255).convert("L")

        if self.combined_transform:
            img, mask = self.combined_transform(img, mask)

        return (img, mask)


class LucchiDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "test"],
        combined_transform: CombinedTransform | None = None,
    ) -> None:
        super().__init__()

        self.root = root
        self.combined_transform = combined_transform
        self.split = split.lower().capitalize()

        self.img_dir = os.path.join(root, f"{self.split}_In")
        self.mask_dir = os.path.join(root, f"{self.split}_Out")

        self.img_files = sorted(
            [f for f in os.listdir(self.img_dir) if f.endswith(".png")]
        )
        self.mask_files = sorted(
            [
                int(f.split(".")[0])
                for f in os.listdir(self.mask_dir)
                if f.endswith(".png")
            ]
        )

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.img_files[idx])).convert("L")
        mask = Image.open(
            os.path.join(self.mask_dir, f"{self.mask_files[idx]}.png")
        ).convert("L")

        if self.combined_transform:
            img, mask = self.combined_transform(img, mask)

        if isinstance(mask, torch.Tensor):
            mask = (mask > 0.5).float()
        else:
            mask = torch.tensor(np.array(mask), dtype=torch.float32)
            mask = (mask > 0.5 * 255).float()

        return (img, mask)
