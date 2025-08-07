import torch
from collections import Counter
from typing import Literal, Union, List, Tuple

from yolo.constants import NUM_BBOXES_PER_SPLIT, NUM_CLASSES, SPLIT_SIZE

Number = Union[float, int]
ListOfListOfNumbers = List[List[Number]]


def intersection_over_union(
    box_1: torch.Tensor,
    box_2: torch.Tensor,
    format: Literal["midpoint", "corner"] = "midpoint",
):
    if format == "midpoint":
        box1_x1 = box_1[..., 0] - box_1[..., 2] / 2
        box1_y1 = box_1[..., 1] - box_1[..., 3] / 2
        box1_x2 = box_1[..., 0] + box_1[..., 2] / 2
        box1_y2 = box_1[..., 1] + box_1[..., 3] / 2

        box2_x1 = box_2[..., 0] - box_2[..., 2] / 2
        box2_y1 = box_2[..., 1] - box_2[..., 3] / 2
        box2_x2 = box_2[..., 0] + box_2[..., 2] / 2
        box2_y2 = box_2[..., 1] + box_2[..., 3] / 2
    elif format == "corner":
        box1_x1, box1_y1, box1_x2, box1_y2 = (
            box_1[..., 0],
            box_1[..., 1],
            box_1[..., 2],
            box_1[..., 3],
        )
        box2_x1, box2_y1, box2_x2, box2_y2 = (
            box_2[..., 0],
            box_2[..., 1],
            box_2[..., 2],
            box_2[..., 3],
        )

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


def format_model_output(
    out: torch.Tensor,
    B: int = NUM_BBOXES_PER_SPLIT,
    S: int = SPLIT_SIZE,
) -> ListOfListOfNumbers:
    boxes = []

    batch_size = out.size(0)

    for b in range(batch_size):
        for i in range(S):
            for j in range(S):
                cell = out[b, i, j]
                class_idx = torch.argmax(cell[:20])

                if B == 1:
                    box = cell[21:25]
                    prob = cell[class_idx] * cell[20]
                else:
                    bbox_score1, bbox_score2 = cell[20], cell[25]

                    if bbox_score1 > bbox_score2:
                        box = cell[21:25]
                        prob = cell[class_idx] * cell[20]
                    else:
                        box = cell[26:30]
                        prob = cell[class_idx] * cell[25]

                x_cell, y_cell = box[0], box[1]  # relative to cell
                width_img, height_img = box[2], box[3]  # relative to image

                x_img = (x_cell + j) / S
                y_img = (y_cell + i) / S

                if prob != 0.0:
                    boxes.append([class_idx, prob, x_img, y_img, width_img, height_img])

    return boxes


def get_bboxes(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: Literal["cuda", "cpu"] = "cuda",
) -> Tuple[ListOfListOfNumbers, ListOfListOfNumbers]:
    model.eval()

    all_gt_bboxes = []
    all_pred_bboxes = []

    train_idx = 0

    for batch_idx, (img, labels) in enumerate(loader):
        img, labels = img.to(device), labels.to(device)

        with torch.no_grad():
            out: torch.Tensor = model(img)

        batch_size = out.shape[0]

        gt_bboxes = format_model_output(labels, 1)
        pred_bboxes = format_model_output(
            out.view(
                batch_size,
                SPLIT_SIZE,
                SPLIT_SIZE,
                NUM_BBOXES_PER_SPLIT * 5 + NUM_CLASSES,
            )
        )

        for idx in range(batch_size):
            train_idx = batch_idx * batch_size + idx

            for gt_box in gt_bboxes:
                all_gt_bboxes.append([train_idx] + gt_box)

            for pred_box in pred_bboxes:
                all_pred_bboxes.append([train_idx] + pred_box)

    return all_gt_bboxes, all_pred_bboxes


def mean_average_precision(
    gt_bboxes: ListOfListOfNumbers,
    pred_bboxes: ListOfListOfNumbers,
    num_classes: int = NUM_CLASSES,
    iou_threshold: Number = 0.5,
):
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        preds = []
        gts = []

        for pred in pred_bboxes:
            if pred[1] == c:
                preds.append(pred)

        for gt in gt_bboxes:
            if gt[1] == c:
                gts.append(gt)

        amount_gt_bboxes_counter = Counter([gt[0] for gt in gts])
        amount_gt_bboxes = {
            img_idx: torch.zeros(num_gts)
            for img_idx, num_gts in amount_gt_bboxes_counter.items()
        }

        preds.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros((len(preds)))
        FP = torch.zeros((len(preds)))

        if len(gts) == 0:
            continue

        for pred_idx, pred in enumerate(preds):
            img_idx = pred[0]
            gts_for_img = [gt for gt in gts if gt[0] == img_idx]

            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gts_for_img):
                iou = intersection_over_union(
                    torch.tensor(pred[3:]), torch.tensor(gt[3:]), format="midpoint"
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                if amount_gt_bboxes[img_idx][best_gt_idx] == 0:
                    TP[pred_idx] = 1
                    amount_gt_bboxes[img_idx][best_gt_idx] = 1
                else:
                    FP[pred_idx] = 1
            else:
                FP[pred_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (len(gts) + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        device = TP.device
        precisions = torch.cat((torch.tensor([1.0], device=device), precisions))
        recalls = torch.cat((torch.tensor([0.0], device=device), recalls))

        average_precisions.append(torch.trapz(precisions, recalls))

    if len(average_precisions) == 0:
        return 0.0

    return sum(average_precisions) / len(average_precisions)
