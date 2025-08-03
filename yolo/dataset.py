import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
from typing import Optional

from yolo.constants import NUM_BBOXES_PER_SPLIT, NUM_CLASSES, SPLIT_SIZE


class VOCDataset(Dataset):
    def __init__(
        self,
        root: str,
        csv_file: str,
        transform,
        img_dir: str = "images",
        label_dir: str = "labels",
        split_size: int = SPLIT_SIZE,
        num_bboxes: int = NUM_BBOXES_PER_SPLIT,
        num_classes: int = NUM_CLASSES,
        header: Optional[int] = None,
    ):
        super().__init__()

        self.root = root
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.df = pd.read_csv(os.path.join(root, csv_file), header=header)
        self.transform = transform

        self.S = split_size
        self.B = num_bboxes
        self.C = num_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, self.img_dir, row.iloc[0])
        label_path = os.path.join(self.root, self.label_dir, row.iloc[1])

        img = Image.open(img_path)
        output = torch.zeros((self.S, self.S, 5 + self.C))

        f = open(label_path)
        lines = f.read().splitlines()

        for line in lines:
            parts = line.split(" ")
            class_idx = int(parts[0])
            bboxes = [float(v) for v in parts[1:]]

            x, y = bboxes[0], bboxes[1]
            i = int(x * self.S)
            j = int(y * self.S)

            class_preds = torch.zeros((self.C))
            class_preds[class_idx] = 1

            class_tensor = torch.zeros(self.C)
            class_tensor[class_idx] = 1.0

            bbox_tensor = torch.tensor([1.0] + bboxes)

            output[i][j] = torch.cat((class_tensor, bbox_tensor))

        return self.transform(img), output
