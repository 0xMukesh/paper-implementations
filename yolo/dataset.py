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
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(val) for val in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])

        if self.transform:
            img = self.transform(img)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5))

        for box in boxes:
            class_label, x, y, width, height = box
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                # `x_cell` and `y_cell` are relative to the grid cell
                # `width` and `height` are relative to the image
                box_coordinates = torch.tensor([x_cell, y_cell, width, height])
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return img, label_matrix
