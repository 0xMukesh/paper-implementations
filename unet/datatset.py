from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import math
from typing import Tuple, Literal


class TeethSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str,
        imgs_dir: str,
        masks_dir: str,
        target_size: Tuple[int, int],
        split: Literal["train", "val"],
        transform=None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.imgs_dir = os.path.join(root, imgs_dir)
        self.masks_dir = os.path.join(root, masks_dir)
        self.target_size = target_size
        self.transform = transform

        self.total_len = len(os.listdir(self.imgs_dir))

        np.random.seed(seed)
        self.img_nums = np.random.choice(
            np.arange(1, self.total_len + 1),
            (
                math.floor(self.total_len * 0.8)
                if split == "train"
                else math.ceil(self.total_len * 0.2)
            ),
            replace=False,
        )

    def __len__(self) -> int:
        return len(self.img_nums)

    def __getitem__(self, idx) -> Tuple[Image.Image, Image.Image]:
        img = Image.open(
            os.path.join(self.imgs_dir, f"{self.img_nums[idx]}.jpg"),
        ).convert("L")
        mask = Image.open(
            os.path.join(self.masks_dir, f"{self.img_nums[idx]}.png")
        ).convert("L")

        img = img.resize(self.target_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.target_size, Image.Resampling.NEAREST)
        mask = mask.point(lambda x: 255 if x > 0 else 0)  # type: ignore

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return (img, mask)
