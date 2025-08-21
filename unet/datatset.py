from torch.utils.data import Dataset
from PIL import Image
import os
from typing import Tuple
import numpy as np


class TeethSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str,
        imgs_dir: str,
        masks_dir: str,
        target_size: Tuple[int, int],
        transform=None,
    ) -> None:
        super().__init__()

        self.imgs_dir = os.path.join(root, imgs_dir)
        self.masks_dir = os.path.join(root, masks_dir)
        self.target_size = target_size
        self.transform = transform

    def __len__(self) -> int:
        return len(os.listdir(self.imgs_dir))

    def __getitem__(self, idx) -> Tuple[Image.Image, Image.Image]:
        img = Image.open(
            os.path.join(self.imgs_dir, f"{idx+1}.jpg"),
        ).convert("L")
        mask = Image.open(os.path.join(self.masks_dir, f"{idx+1}.png")).convert("L")

        # resize image using bilinear for smooth gradient
        img = img.resize(self.target_size, Image.Resampling.BILINEAR)
        # resize mask using nearset to avoid creating additional class values, as bilinear uses intermediate values
        mask = mask.resize(self.target_size, Image.Resampling.NEAREST)

        # convert mask to binary image/matrix which contains either 1 (if tooth exists) or 0
        mask_img_array = np.array(mask)
        binary_array = np.where(mask_img_array > 0, 255, 0).astype(np.uint8)
        mask = Image.fromarray(binary_array, mode="L")

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return (img, mask)
