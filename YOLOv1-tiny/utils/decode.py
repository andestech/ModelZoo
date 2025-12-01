from typing import List, Tuple
import numpy as np
import torch
from torch import Tensor

__all__ = ["decode_v1"]


def decode_v1(pred_tensor: Tensor, num_classes: int = 20, prob_threshold: float = 0.01):
    """Decode tensor into box coordinates, class labels, and probs_detected.
    Args:
        pred_tensor: (tensor) tensor to decode sized [S, S, (5xB+C)], 5=(x_c, y_c, w, h, conf)
    Returns:
        boxes: (tensor) [[x_c, y_c, w, h], ...]. Normalized from 0.0 to 1.0 w.r.t. image width/height, sized [n_boxes, 4].
        confidences: (tensor) objectness confidences for each detected box, sized [n_boxes].
        scores: (tensor) scores for most likely class for each detected box, sized [n_boxes].
        labels: (tensor) class labels for each detected boxe, sized [n_boxes].
    """
    pred_tensor = pred_tensor.squeeze(0)
    device = pred_tensor.device
    num_bboxes = int((pred_tensor.size(-1) - num_classes) // 5)
    num_grids = int(pred_tensor.size(0))
    conf_idx = torch.arange(4, 5 * num_bboxes, 5)
    box_idx = np.arange(5 * num_bboxes)
    box_idx = torch.tensor(np.delete(box_idx, conf_idx), device=device)
    conf_idx = conf_idx.to(device)
    x_idx = torch.arange(0, len(box_idx), 4, device=device)
    y_idx = torch.arange(1, len(box_idx), 4, device=device)
    w_idx = torch.arange(2, len(box_idx), 4, device=device)
    h_idx = torch.arange(3, len(box_idx), 4, device=device)

    # [S, S, 4 * B]
    boxes = pred_tensor.index_select(dim=-1, index=box_idx)
    corner_y = torch.arange(num_grids, dtype=boxes.dtype, device=device)
    corner_y = corner_y.view(-1, 1, 1).expand((num_grids, num_grids, num_bboxes))
    corner_x = corner_y.transpose(0, 1)
    boxes[..., x_idx] = (boxes[..., x_idx] + corner_x) / num_grids
    boxes[..., y_idx] = (boxes[..., y_idx] + corner_y) / num_grids
    boxes[..., w_idx] = boxes[..., w_idx].pow(2)
    boxes[..., h_idx] = boxes[..., h_idx].pow(2)

    # [S, S, B]
    confidences = pred_tensor.index_select(dim=-1, index=conf_idx)
    # [S, S, 1], [S, S, 1]
    scores, labels = torch.max(pred_tensor[..., 5 * num_bboxes :], dim=-1, keepdim=True)
    # [S, S, 1] -> [S, S, B]
    scores = scores.repeat_interleave(num_bboxes, dim=-1)
    labels = labels.repeat_interleave(num_bboxes, dim=-1)

    # filter
    mask = confidences * scores > prob_threshold  # [S, S, B]
    boxes = boxes[mask.repeat_interleave(4, dim=-1)].view(-1, 4)
    confidences = confidences[mask]
    scores = scores[mask]
    labels = labels[mask]
    return boxes, confidences, scores, labels
