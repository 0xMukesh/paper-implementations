import torch
from torch import nn

from yolo.constants import NUM_BBOXES_PER_SPLIT, NUM_CLASSES, SPLIT_SIZE
from yolo.utils import intersection_over_union

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")

        self.split_size = SPLIT_SIZE
        self.num_bboxes = NUM_BBOXES_PER_SPLIT
        self.num_classes = NUM_CLASSES
        self.epsilon = 1e-6

        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        predictions = predictions.reshape(
            -1,
            self.split_size,
            self.split_size,
            self.num_classes + self.num_bboxes * 5
        )

        I_obj = targets[..., 20]               
        I_noobj = 1 - I_obj
        target_bbox_coords = targets[..., 21:25] 

        pred_bbox1_coords = predictions[..., 21:25]
        pred_bbox2_coords = predictions[..., 26:30]

        iou_bbox1 = intersection_over_union(pred_bbox1_coords, target_bbox_coords)
        iou_bbox2 = intersection_over_union(pred_bbox2_coords, target_bbox_coords)
        ious = torch.cat((iou_bbox1.unsqueeze(0), iou_bbox2.unsqueeze(0)))

        _, bestbox = torch.max(ious, dim=0)
        bestbox = bestbox.float()

        box_preds = bestbox.unsqueeze(-1) * pred_bbox2_coords + (1 - bestbox.unsqueeze(-1)) * pred_bbox1_coords
        box_preds = I_obj.unsqueeze(-1) * box_preds
        box_targets = I_obj.unsqueeze(-1) * target_bbox_coords

        box_preds_wh = torch.sqrt(torch.clamp(box_preds[..., 2:4], min=self.epsilon))
        box_preds_modified = torch.cat([box_preds[..., :2], box_preds_wh], dim=-1)

        box_targets_wh = torch.sqrt(torch.clamp(box_targets[..., 2:4], min=self.epsilon))
        box_targets_modified = torch.cat([box_targets[..., :2], box_targets_wh], dim=-1)

        box_loss = self.mse(box_preds_modified, box_targets_modified)

        pred_confidence1 = predictions[..., 20]
        pred_confidence2 = predictions[..., 25]
        pred_confidence = bestbox * pred_confidence2 + (1 - bestbox) * pred_confidence1
        pred_confidence = I_obj * pred_confidence
        target_confidence = I_obj * targets[..., 20]
        obj_loss = self.mse(pred_confidence.flatten(), target_confidence.flatten())

        no_obj_loss = self.mse(
            (I_noobj * pred_confidence1).flatten(),
            (I_noobj * targets[..., 20]).flatten()
        )
        no_obj_loss += self.mse(
            (I_noobj * pred_confidence2).flatten(),
            (I_noobj * targets[..., 20]).flatten()
        )

        pred_class_probs = predictions[..., :self.num_classes].flatten(end_dim=-2)
        target_class_probs = targets[..., :self.num_classes].flatten(end_dim=-2)
        class_loss = self.mse(pred_class_probs, target_class_probs)

        loss = (
            self.lambda_coord * box_loss
            + obj_loss
            + self.lambda_noobj * no_obj_loss
            + class_loss
        )
        return loss
