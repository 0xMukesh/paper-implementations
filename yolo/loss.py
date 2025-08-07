import torch
from torch import nn

from yolo.constants import NUM_BBOXES_PER_SPLIT, NUM_CLASSES, SPLIT_SIZE
from yolo.utils import intersection_over_union


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = SPLIT_SIZE
        self.B = NUM_BBOXES_PER_SPLIT
        self.C = NUM_CLASSES
        self.epsilon = 1e-6
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        target_classes = targets[..., : self.C]
        target_confidence = targets[..., self.C : self.C + 1]
        target_boxes = targets[..., self.C + 1 : self.C + 5]

        pred_classes = predictions[..., : self.C]
        pred_confidence1 = predictions[..., self.C : self.C + 1]
        pred_boxes1 = predictions[..., self.C + 1 : self.C + 5]
        pred_confidence2 = predictions[..., self.C + 5 : self.C + 6]
        pred_boxes2 = predictions[..., self.C + 6 : self.C + 10]

        iou_b1 = intersection_over_union(pred_boxes1, target_boxes, format="midpoint")
        iou_b2 = intersection_over_union(pred_boxes2, target_boxes, format="midpoint")

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        _, bestbox = torch.max(ious, dim=0)

        exists_box = target_confidence

        bestbox_expanded = bestbox.unsqueeze(3)

        box_predictions = exists_box * (
            (1 - bestbox_expanded) * pred_boxes1 + bestbox_expanded * pred_boxes2
        )
        box_targets = exists_box * target_boxes

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + self.epsilon)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4] + self.epsilon)

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        pred_confidence = (
            1 - bestbox_expanded
        ) * pred_confidence1 + bestbox_expanded * pred_confidence2

        obj_loss = self.mse(
            torch.flatten(exists_box * pred_confidence),
            torch.flatten(exists_box * target_confidence),
        )

        no_obj_loss = self.mse(
            torch.flatten((1 - exists_box) * pred_confidence1),
            torch.flatten((1 - exists_box) * torch.zeros_like(target_confidence)),
        )
        no_obj_loss += self.mse(
            torch.flatten((1 - exists_box) * pred_confidence2),
            torch.flatten((1 - exists_box) * torch.zeros_like(target_confidence)),
        )

        class_loss = self.mse(
            torch.flatten(exists_box * pred_classes, end_dim=-2),
            torch.flatten(exists_box * target_classes, end_dim=-2),
        )

        loss = (
            self.lambda_coord * box_loss
            + obj_loss
            + self.lambda_noobj * no_obj_loss
            + class_loss
        )

        return loss, box_loss, obj_loss, no_obj_loss, class_loss
