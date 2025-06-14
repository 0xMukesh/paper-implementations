import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

def tensor_to_cv_img(tensor: Tensor):
    tensor = tensor.squeeze(0) # remove batch dim
    # [3, H, W] -> [H, W, 3]
    img = tensor.permute(1, 2, 0).numpy()
    
    if img.max() <= 1.0:
        img = (img * np.array(255)).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def cv_img_to_tensor(img):
    # h, w, c -> c, h, w
    img = np.permute_dims(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float)

def selective_search(img, maxBboxes):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()

    bboxes = ss.process()
    filteredBboxes = [bbox for bbox in bboxes if bbox[2] >= 10 and bbox[3] >= 10]

    return filteredBboxes[:maxBboxes]

def label_str_to_one_hot(str: str, labels: list[str]):
    for i, s in enumerate(labels):
        if s == str:
            return F.one_hot(torch.tensor(i), len(labels))
        else:
            return -1

def compute_iou(box_a: list[int], box_b: list[int]):
    x_min = max(box_a[0], box_b[0])
    y_min = max(box_a[1], box_b[1])
    x_max = min(box_a[2], box_b[2])
    y_max = min(box_a[3], box_b[3])

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    inter_area = max(0, x_max - x_min) * max(0, y_max - y_min)
    union_area = box_a_area + box_b_area - inter_area

    return inter_area / union_area if union_area != 0 else 0
