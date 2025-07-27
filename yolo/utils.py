import torch

def intersection_over_union(box_1: torch.Tensor, box_2: torch.Tensor, format: str = "midpoint"):
    if format == "midpoint":
        box1_x1 = box_1[..., 0] - box_1[..., 2] / 2
        box1_y1 = box_1[..., 1] - box_1[..., 3] / 2
        box1_x2 = box_1[..., 0] + box_1[..., 2] / 2
        box1_y2 = box_1[..., 1] + box_1[..., 3] / 2

        box2_x1 = box_2[..., 0] - box_2[..., 2] / 2
        box2_y1 = box_2[..., 1] - box_2[..., 3] / 2
        box2_x2 = box_2[..., 0] + box_2[..., 2] / 2
        box2_y2 = box_2[..., 1] + box_2[..., 3] / 2
    else:
        box1_x1, box1_y1, box1_x2, box1_y2 = box_1[..., 0], box_1[..., 1], box_1[..., 2], box_1[..., 3]
        box2_x1, box2_y1, box2_x2, box2_y2 = box_2[..., 0], box_2[..., 1], box_2[..., 2], box_2[..., 3]

    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou
