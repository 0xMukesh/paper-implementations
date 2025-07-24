import cv2
import torch
from typing import Sequence
from cv2.typing import Rect
from collections import Counter

"""
In R-CNN,

i) Selective search is used to generate ~2k region proposals
ii) IoU (intersection over union) is applied on each of these region proposals to find the appropriate GT values
iii) The region proposals are resized and passed into the convolutional layers
iv) The output of the convolutional layers is then passed into the classification head and regression head
v) Cross entropy loss is used in the classification head and L2 loss is used in the regression head and their weighted sum acts like the total loss for the network
vi) Within the regression head, instead of determing the coordinates of the bounding box from scratch, it tries to predict the delta (transform)

g_x = p_x + p_w * t_x => t_x = (g_x - p_x)/p_w
g_y = p_y + p_h * t_y => t_y = (g_y - p_y)/p_h
g_w = p_w * exp(t_w) => t_w = log(g_w/p_w)
g_h = p_h * exp(t_h) => t_h = log(g_h/p_h)

The regression head would tries to learn t_x, t_y, t_w and t_h values.

vii) When there are multiple bounding boxes pointing to the same object, NMS (non-maximum suppression) is applied (during post-processing) to remove the redutant bounding boxes.
viii) mAP (mean average precision) is used evalute the object detection model.
"""

Bbox = tuple[float, float, float, float]

def rect_to_bbox(r: Rect) -> Bbox:
    return (r[0], r[1], r[2], r[3])

def selective_search(img_path: str, num_region_proposals: int):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects[:num_region_proposals]
    
def compute_iou(box1: Bbox, box2: Bbox):
    (x1, y1, w1, h1) = box1
    (x2, y2, w2, h2) = box2

    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1+w1, x2+w2)
    yB = min(y1+h1, y2+h2)

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    
    intersection_area = inter_width * inter_height
    union_area = (w1 * h1) + (w2 * h2) - intersection_area

    if union_area == 0:
        return 0

    return intersection_area / union_area

def apply_iou_to_ss_output(rects: Sequence[Rect], gt: dict[Bbox, str]):
    output: dict[Bbox, tuple[Bbox, str]] = {}

    for r in rects:
        r = rect_to_bbox(r)

        best_iou = 0.0
        best_label = None
        best_bbox = None

        for bbox, label in gt.items():
            iou = compute_iou(r, bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_label = label
                best_bbox = bbox

        if best_label is not None and best_bbox is not None:
            if best_iou >= 0.5:
                output[r] = (best_bbox, best_label)
            elif 0.3 <= best_iou < 0.5:
                output[r] = (best_bbox, "__background__")

    return output


def apply_nms(predictions: list[tuple[int, float, float, float, float, float]], prob_threshold = 0.2, iou_threshold = 0.5):
    # predictions - [[class, probability, x1, y1, x2, y2]]

    boxes = [box for box in predictions if box[0] > prob_threshold]
    boxes = sorted(boxes, key=lambda x: x[0], reverse=True)
    output = []

    while boxes:
        chosen_box = boxes.pop(0)
        boxes = [
            box for box in boxes if box[0] != chosen_box[0] or compute_iou(box[2:], chosen_box[2:]) < iou_threshold
        ]
        output.append(chosen_box)

    return output

def compute_map(pred_bboxes, gt_bboxes, iou_threshold = 0.5, prob_threshold = 0.2, num_classes = 20):
    # pred_bboxes: [[test_img_idx, class_pred, pred_score, x1, x2, y1, y2], ...]
    # gt_bboxes: [[test_img_idx, class_idx, x1, x2, y1, y2]]
    
    average_precisions: list[torch.Tensor] = [] 

    for c in range(num_classes):
        detections = [] 
        ground_truths = []

        for pred in pred_bboxes:
            if pred[1] == c and pred[2] >= prob_threshold:
                detections.append(pred)

        for gt in gt_bboxes:
            if gt[1] == c:
                ground_truths.append(gt)

        counts = Counter([gt[0] for gt in ground_truths])
        num_bboxes = {k: torch.zeros(v) for k, v in counts.items()}

        detections.sort(key=lambda x: x[2], reverse=True) # sort with decreasing order of predicted score

        true_positives = torch.zeros(len(detections))
        false_positives = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        for (detection_idx, detection) in enumerate(detections):
            gts = [gt for gt in ground_truths if gt[0] == detection[0]]
            best_iou = 0
            best_gt_idx = -1

            for (gt_idx, gt) in enumerate(gts):
                gt_bbox = gt[2:]
                pred_bbox = detection[3:]

                iou = compute_iou(gt_bbox, pred_bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou > iou_threshold:
                if num_bboxes[detection[0]][best_gt_idx] == 0:
                    true_positives[detection_idx] = 1
                    num_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    false_positives[detection_idx] = 1
            else:
                false_positives[detection_idx] = 1

        true_positives_cumsum = torch.cumsum(true_positives, dim=0)
        false_positive_cumsum = torch.cumsum(false_positives, dim=0)

        precision = torch.divide(true_positives_cumsum, true_positives_cumsum + false_positive_cumsum)
        recall = torch.divide(true_positives_cumsum, total_true_bboxes)
        # adds the "initial" trapezoid while calculating AUC
        precision = torch.cat((torch.tensor([1]), precision), dim=0) 
        recall = torch.cat((torch.tensor([0]), recall), dim=0)

        ap = torch.trapezoid(precision, recall)
        average_precisions.append(ap)

    return sum(average_precisions) / len(average_precisions)

def compute_map_range(pred_bboxes, gt_bboxes, prob_threshold, num_classes, start_iou, stop_iou, step_size):
    total_map = 0
    num_ious = 0

    for iou in range(start_iou, stop_iou, step_size):
        total_map += compute_map(pred_bboxes, gt_bboxes, iou, prob_threshold, num_classes)
        num_ious += 1

    return total_map / num_ious

