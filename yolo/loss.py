import torch
from torch import nn

from .utils import intersection_over_union
from .constants import SPLIT_SIZE, NUM_BBOXES_PER_SPLIT, NUM_CLASSES

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss(reduction="sum")

        self.split_size = SPLIT_SIZE
        self.num_bboxes = NUM_BBOXES_PER_SPLIT
        self.num_classes = NUM_CLASSES
        self.epsilon = 1e-6

        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predications: torch.Tensor, targets: torch.Tensor):
        # targets: (B, S, S, C + 5)
        # predications: (B, S, S, C + 5 * B)
        predications = predications.reshape(-1, self.split_size, self.split_size, self.num_classes + self.num_bboxes * 5)

        target_bbox_coords = targets[..., 21:25]
        I_obj = targets[..., 20]
        I_noobj = 1 - I_obj

        pred_bbox1_coords = predications[..., 21:25]
        pred_bbox2_coords = predications[..., 26:30]

        # iou_bbox1/iou_bbox2: (B, S, S)
        iou_bbox1 = intersection_over_union(pred_bbox1_coords, target_bbox_coords) 
        iou_bbox2 = intersection_over_union(pred_bbox2_coords, target_bbox_coords)

        # ious: (2, B, S, S)
        ious = torch.cat((iou_bbox1.unsqueeze(0), iou_bbox2.unsqueeze(0)))

        # bestbox: (B, S, S)
        # bestbox contains either `0` or `1` depending on which bbox (bbox1 or bbox2``) is the best suit for that split
        _, bestbox = torch.max(ious, dim=0)
        bestbox = bestbox.float()

        # 1. calculating bounding box loss
        # bbox loss is penalized only if an object exists in that cell
        box_preds = I_obj.unsqueeze(-1) * (bestbox.unsqueeze(-1) * pred_bbox2_coords + (1 - bestbox.unsqueeze(-1)) * pred_bbox1_coords)
        box_targets = I_obj.unsqueeze(-1) * target_bbox_coords

        # taking sqrt for weight and height to penalize error more when weight and height are small
        box_preds[..., 2:4] = torch.sign(box_preds[..., 2:4]) * torch.sqrt(box_preds[..., 2:4].abs() + self.epsilon)
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4].abs())

        # box_preds/box_targets: (B, S, S, 4) -> (B * S * S, 4)
        box_preds = box_preds.flatten(end_dim=-2)
        box_targets = box_targets.flatten(end_dim=-2)

        box_loss = self.mse(box_preds, box_targets)

        # 2. calculating object loss
        # object loss is penalized only if an object exists in that cell
        pred_bbox1_obj_exists = predications[..., 20]
        pred_bbox2_obj_exists = predications[..., 25]
        target_obj_exists = targets[..., 20]

        pred_obj_exists = I_obj * (bestbox * pred_bbox2_obj_exists + (1 - bestbox) * pred_bbox1_obj_exists)
        target_obj_exists = I_obj * target_obj_exists
        
        obj_loss = self.mse(pred_obj_exists.flatten(), target_obj_exists.flatten())

        # 3. calculating no object loss
        # no object loss is always penalized
        no_obj_loss = self.mse((I_noobj * pred_bbox1_obj_exists).flatten(), (I_noobj * target_obj_exists).flatten())
        no_obj_loss += self.mse((I_noobj * pred_bbox2_obj_exists).flatten(), (I_noobj * target_obj_exists).flatten())

        # 4. calculating class loss
        pred_class_probs = predications[..., :20].flatten(end_dim=-2)
        target_class_probs = targets[..., :20].flatten(end_dim=-2)

        class_loss = self.mse(pred_class_probs, target_class_probs)

        # 5. calculating final total loss
        loss = self.lambda_coord * box_loss + obj_loss + self.lambda_noobj * no_obj_loss + class_loss
        return loss
