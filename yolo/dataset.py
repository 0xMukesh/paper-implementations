from typing import Tuple, TypeVar, Callable, Generic
from PIL.Image import Image
from torch.utils.data import Dataset
import torch
from torchvision import datasets

from yolo.constants import NUM_CLASSES, SPLIT_SIZE, VOC_CLASSES

VOC_CLASSES_TO_INDEX = {cls: index for index, cls in enumerate(VOC_CLASSES)}

T = TypeVar("T")

class VOCDataset(Dataset, Generic[T]):
    def __init__(self, root: str, year: str, image_set: str, download: bool, transform: Callable[[Image], T]):
        super().__init__()

        self.split_size = SPLIT_SIZE
        self.num_classes = NUM_CLASSES
        self.dataset = datasets.VOCDetection(root, year, image_set, download)
        self.transform = transform

    def __getitem__(self, index) -> Tuple[T, torch.Tensor]:
        img, annotations = self.dataset.__getitem__(index)
        output = torch.zeros((self.split_size, self.split_size, self.num_classes + 5))

        w = int(annotations["annotation"]["size"]["width"])
        h = int(annotations["annotation"]["size"]["height"])

        for obj in annotations["annotation"]["object"]:
            obj_idx = VOC_CLASSES_TO_INDEX[obj["name"]]
            xmin = int(obj["bndbox"]["xmin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymin = int(obj["bndbox"]["ymin"])
            ymax = int(obj["bndbox"]["ymax"])

            x_center = ((xmax + xmin) / 2) / w
            y_center = ((ymax + ymin) / 2) / h
            bbox_w = (xmax - xmin) / w
            bbox_h = (ymax - ymin) / h

            i = int(y_center * self.split_size)
            j = int(x_center * self.split_size)

            onehot = torch.zeros(self.num_classes)
            onehot[obj_idx] = 1
            bbox = torch.tensor([1.0, x_center, y_center, bbox_w, bbox_h])
            output[i, j] = torch.cat([onehot, bbox])

        return self.transform(img), output

    def __len__(self) -> int:
        return len(self.dataset)