from typing import Dict

import numpy as np
from torch import Tensor, bool, cat, tensor, zeros
from torch.nn.functional import one_hot

from .utils import compute_iou

__all__ = [
    "calculate_AP",
    "calculate_all_AP",
    "calculate_voc2007_AP",
    "evaluate_map",
    "evaluate_map_voc",
]


def calculate_AP(recall: np.ndarray, precision: np.ndarray) -> float:
    """Calculate AP for one class based on VOC 2012 dataset criterion.
    Args:
        recall (np.ndarray): recall values of precision-recall curve.
        precision (np.ndarray): precision values of precision-recall curve.
    Returns:
        average precision (float): average precision (AP) for each class.
    """
    # AP (AUC of precision-recall curve) computation using all points interpolation.
    # For mAP computation, you can find a great explaination below.
    # https://github.com/rafaelpadilla/Object-Detection-Metrics

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    ap = 0.0  # average precision (AUC of the precision-recall curve).
    for i in range(precision.size - 1):
        ap += (recall[i + 1] - recall[i]) * precision[i + 1]
    return ap


def calculate_all_AP(recall: Tensor, precision: Tensor) -> Tensor:
    """Calculate AP for all class based on VOC 2012 dataset criterion.
    Args:
        recall (np.ndarray): recall values of precision-recall curve.
        precision (np.ndarray): precision values of precision-recall curve.
    Returns:
        average precision (float): average precision (AP) for all class.
    """
    sorted_recall, sorted_idx = recall.sort(dim=0, descending=True)
    sorted_precision = precision.gather(dim=0, index=sorted_idx)

    delta_x = sorted_recall[:-1] - sorted_recall[1:]
    sorted_precision, _ = sorted_precision.cummax(dim=0)
    aps = (delta_x * sorted_precision[:-1]).sum(dim=0)
    return aps


def calculate_voc2007_AP(recall: np.ndarray, precision: np.ndarray) -> float:
    """Calculate AP for one class based on VOC 2007 dataset criterion.
    Args:
        recall (np.ndarray): recall values of precision-recall curve.
        precision (np.ndarray): precision values of precision-recall curve.
    Returns:
        average precision (float): average precision (AP) for each class.
    """
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.0
    return ap


def evaluate_map(
    pred_bboxes: Tensor,
    pred_classes: Tensor,
    pred_class_scores: Tensor,
    pred_ids: Tensor,
    target_bboxes: Tensor,
    target_classes: Tensor,
    target_ids: Tensor,
    class_names: list,
    iou_threshold: float = 0.5,
    x1y1x2y2: bool = False,
) -> Tensor:
    """
    Args:
        pred_bboxes (Tensor): detected boxes, size [n_bboxes_1, 4]
        pred_classes (Tensor): predicted class for each box, size [n_bboxes_1]
        pred_class_scores (Tensor): predicted score for predict class, size [n_bboxes_1]
        pred_ids (Tensor): images order, size [n_bboxes_1]
        target_bboxes (Tensor): ground truth boxes, size [n_bboxes_2, 4]
        target_classes (Tensor): ground truth cate categories, size [n_bboxes_2]
        target_ids (Tensor): images order, size [n_bboxes_2]
        class_list (list): class names in a list
        iou_threshold (float): if iou less than this threshold is FP, otherwise TP
        x1y1x2y2 (bool): if using x1y1x2y2 coordination set True, cxcywh set False
    Return:
        aps (Tensor): mean average precision of each class
    """
    assert (
        len(pred_bboxes.shape) == len(target_bboxes.shape) == 2
    ), f"Predicted bounding boxes {pred_bboxes.shape} and target bounding boxes {target_bboxes.shape} have same dimensions."
    assert (
        pred_bboxes.shape[-1] == target_bboxes.shape[-1] == 4
    ), f"Predicted bounding boxes {pred_bboxes.shape} and target bounding boxes {target_bboxes.shape} have 4 features (x_c, y_c, w, h)."
    device = pred_bboxes.device
    if pred_classes.device != device:
        pred_classes = pred_classes.to(device)
    if pred_class_scores.device != device:
        pred_class_scores = pred_class_scores.to(device)
    if pred_ids.device != device:
        pred_ids = pred_ids.to(device)
    if target_classes.device != device:
        target_classes = target_classes.to(device)
    if target_ids.device != device:
        target_ids = target_ids.to(device)
    if pred_classes.dim() == 1:
        pred_classes = one_hot(pred_classes.long(), num_classes=len(class_names))
    if target_classes.dim() == 1:
        target_classes = one_hot(target_classes.long(), num_classes=len(class_names))

    TP = []
    for id in pred_ids.unique():
        # pick this image related bboxes
        id_mask: Tensor = pred_ids == id
        target_id_mask: Tensor = target_ids == id

        _pred_bboxes = pred_bboxes[id_mask]
        _pred_classes = pred_classes[id_mask]
        _pred_class_scores = pred_class_scores[id_mask]
        _target_bboxes = target_bboxes[target_id_mask]
        _target_classes = target_classes[target_id_mask]

        # [n_bboxes_1, n_bboxes_2, n_classes]
        class_mask = _pred_classes.unsqueeze(1) * _target_classes.unsqueeze(0)
        idx3d = (_pred_class_scores.view(-1, 1, 1) * class_mask).argsort(dim=0, descending=True)
        reverse_idx3d = idx3d.argsort(0)
        # filter out iou less than threshold, [n_bboxes_1, n_bboxes_2, n_classes]
        ious = compute_iou(_pred_bboxes, _target_bboxes, compare_all=True, x1y1x2y2=x1y1x2y2)
        ious = ious.unsqueeze(-1) * class_mask
        # every detected boxes pick the largest ious from "all the ground truths"
        ious *= ious == ious.amax(dim=1, keepdim=True)
        # sort the orthogonal_ious by confidence and pick the first True value
        ious = ious.gather(0, idx3d) >= iou_threshold
        first_true_idx = ious.int().argmax(0, keepdim=True)
        max_mask = zeros(ious.shape, dtype=bool, device=ious.device)
        max_mask[first_true_idx] = True
        ious *= max_mask
        # reverse sort and filter out all zero columns, [n_bboxes_1, n_classes]
        ious = ious.gather(dim=0, index=reverse_idx3d).sum(1).bool()
        TP.append(ious)

    # [n_bboxes_1, n_classes]
    pred_each_class_scores = pred_class_scores.unsqueeze(-1) * pred_classes
    sorted_each_class_scores, idx = pred_each_class_scores.sort(dim=0, descending=True)

    # [n_bboxes_1, n_classes]
    TP = cat(TP).gather(dim=0, index=idx).cumsum(dim=0)
    all_detections = sorted_each_class_scores.bool().cumsum(dim=0)

    # [1, n_classes]
    all_ground_truth = target_classes.bool().sum(dim=0, keepdim=True)

    # [n_bboxes_1, n_classes]
    recalls = (TP / all_ground_truth).nan_to_num()
    precisions = (TP / all_detections).nan_to_num()
    nonzero_idx = (recalls + precisions).sum(dim=1) != 0
    aps = calculate_all_AP(recalls[nonzero_idx], precisions[nonzero_idx])
    return aps

def evaluate_map_voc(
    pred_bboxes: Tensor,
    pred_classes: Tensor,
    pred_class_scores: Tensor,
    pred_ids: Tensor,
    target_bboxes: Tensor,
    target_classes: Tensor,
    target_difficults: Tensor,
    target_ids: Tensor,
    class_names: list,
    iou_threshold: float = 0.5,
    x1y1x2y2: bool = False,
) -> Tensor:
    """
    Args:
        pred_bboxes (Tensor): detected boxes, size [n_bboxes_1, 4]
        pred_classes (Tensor): predicted class for each box, size [n_bboxes_1]
        pred_class_scores (Tensor): predicted score for predict class, size [n_bboxes_1]
        pred_ids (Tensor): images order, size [n_bboxes_1]
        target_bboxes (Tensor): ground truth boxes, size [n_bboxes_2, 4]
        target_classes (Tensor): ground truth cate categories, size [n_bboxes_2]
        target_difficults (Tensor): difficult flags in voc annotation, size [n_bboxes_2]
        target_ids (Tensor): images order, size [n_bboxes_2]
        class_list (list): class names in a list
        iou_threshold (float): if iou less than this threshold is FP, otherwise TP
        x1y1x2y2 (bool): if using x1y1x2y2 coordination set True, cxcywh set False
    Return:
        aps (Tensor): mean average precision of each class
    """
    assert (
        len(pred_bboxes.shape) == len(target_bboxes.shape) == 2
    ), f"Predicted bounding boxes {pred_bboxes.shape} and target bounding boxes {target_bboxes.shape} have same dimensions."
    assert (
        pred_bboxes.shape[-1] == target_bboxes.shape[-1] == 4
    ), f"Predicted bounding boxes {pred_bboxes.shape} and target bounding boxes {target_bboxes.shape} have 4 features (x_c, y_c, w, h)."
    device = pred_bboxes.device
    if pred_classes.device != device:
        pred_classes = pred_classes.to(device)
    if pred_class_scores.device != device:
        pred_class_scores = pred_class_scores.to(device)
    if pred_ids.device != device:
        pred_ids = pred_ids.to(device)
    if target_classes.device != device:
        target_classes = target_classes.to(device)
    if target_difficults.device != device:
        target_difficults = target_difficults.to(device)
    if target_ids.device != device:
        target_ids = target_ids.to(device)
    if pred_classes.dim() == 1:
        pred_classes = one_hot(pred_classes.long(), num_classes=len(class_names))
    if target_classes.dim() == 1:
        target_classes = one_hot(target_classes.long(), num_classes=len(class_names))
    TP = []
    easy_pred_class_scores = []
    easy_pred_classes = []
    for id in pred_ids.unique():
        # pick this image related bboxes
        id_mask: Tensor = pred_ids == id
        target_id_mask: Tensor = target_ids == id

        _pred_bboxes = pred_bboxes[id_mask]
        _pred_classes = pred_classes[id_mask]
        _pred_class_scores = pred_class_scores[id_mask]
        _target_bboxes = target_bboxes[target_id_mask]
        _target_classes = target_classes[target_id_mask]
        _target_difficults = target_difficults[target_id_mask]

        # [n_bboxes_1, n_bboxes_2, n_classes]
        class_mask = _pred_classes.unsqueeze(1) * _target_classes.unsqueeze(0)
        # filter out iou less than threshold, [n_bboxes_1, n_bboxes_2, n_classes]
        ious = compute_iou(_pred_bboxes, _target_bboxes, compare_all=True, x1y1x2y2=x1y1x2y2)
        ious = ious.unsqueeze(-1) * class_mask
        # every detected boxes pick the largest ious from "all the ground truths"
        ious *= ious == ious.amax(dim=1, keepdim=True)
        # remove difficult cases from all detections and TPs
        detect_difficult_bboxes = ious.sum(dim=-1)[:, _target_difficults.bool()].sum(dim=-1)
        easy_idx = ~detect_difficult_bboxes.bool()
        easy_pred_class_scores.append(_pred_class_scores[easy_idx])
        easy_pred_classes.append(_pred_classes[easy_idx])
        ious = ious[easy_idx]
        if len(ious) > 0:
            # sort the orthogonal_ious by confidence and pick the first True value
            class_mask = _pred_classes[easy_idx].unsqueeze(1) * _target_classes.unsqueeze(0)
            idx3d = (_pred_class_scores[easy_idx].view(-1, 1, 1) * class_mask).argsort(dim=0, descending=True)
            reverse_idx3d = idx3d.argsort(0)
            ious = ious.gather(0, idx3d) >= iou_threshold
            first_true_idx = ious.int().argmax(0, keepdim=True)
            max_mask = zeros(ious.shape, dtype=bool, device=ious.device)
            max_mask[first_true_idx] = True
            ious *= max_mask
            # reverse sort and filter out all zero columns, [n_bboxes_1, n_classes]
            ious = ious.gather(dim=0, index=reverse_idx3d).sum(1).bool()
            TP.append(ious)


    # [n_bboxes_1 - n_difficults, n_classes]
    pred_each_class_scores = cat(easy_pred_class_scores).unsqueeze(-1) * cat(easy_pred_classes)
    sorted_each_class_scores, idx = pred_each_class_scores.sort(dim=0, descending=True)

    # [n_bboxes_1 - n_difficults, n_classes]
    TP = cat(TP).gather(dim=0, index=idx).cumsum(dim=0)
    all_detections = sorted_each_class_scores.bool().cumsum(dim=0)

    # [1, n_classes]
    target_difficults = (target_difficults.unsqueeze(-1) * target_classes).sum(dim=0, keepdim=True)
    all_ground_truth = target_classes.bool().sum(dim=0, keepdim=True) - target_difficults

    # [n_bboxes_1 - n_difficults, n_classes]
    recalls = (TP / all_ground_truth).nan_to_num()
    precisions = (TP / all_detections).nan_to_num()
    nonzero_idx = (recalls + precisions).sum(dim=1) != 0
    aps = calculate_all_AP(recalls[nonzero_idx], precisions[nonzero_idx])
    return aps
