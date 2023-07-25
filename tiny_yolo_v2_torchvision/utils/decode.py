from typing import List
import torch
from torch import Tensor

__all__ = ["decode_v2"]


def decode_v2(
    pred_tensor: Tensor,
    anchors: List[float] = [
        (1.08, 1.19),
        (3.42, 4.41),
        (6.63, 11.38),
        (9.42, 5.11),
        (16.62, 10.52),
    ],
    num_classes: int = 20,
    prob_threshold: float = 0.01,
):
    """Decode tensor into box coordinates, class labels, and probs_detected.
    Args:
        pred_tensor: (tensor) tensor to decode sized [S, S, Bx(5+C)], 5=(x_c, y_c, w, h, conf)
    Returns:
        boxes: (tensor) [[x_c, y_c, w, h], ...]. Normalized from 0.0 to 1.0 w.r.t. image width/height, sized [n_boxes, 4].
        confidences: (tensor) objectness confidences for each detected box, sized [n_boxes].
        scores: (tensor) scores for most likely class for each detected box, sized [n_boxes].
        labels: (tensor) class labels for each detected boxe, sized [n_boxes].
    """
    # preparation
    pred_tensor = pred_tensor.squeeze(0)
    device = pred_tensor.device
    step = int(5 + num_classes)
    num_bboxes = int(pred_tensor.size(-1) // (5 + num_classes))
    num_grids = int(pred_tensor.size(0))
    anchors = torch.tensor(anchors, dtype=pred_tensor.dtype, device=device)

    # idx
    box_idx = torch.cat([torch.arange(i * step, i * step + 4) for i in range(num_bboxes)]).to(device)
    conf_idx = torch.arange(4, num_bboxes * step, step).to(device)
    class_idx = torch.cat([torch.arange(5 + i * step, 5 + i * step + num_classes) for i in range(num_bboxes)]).to(device)
    x_idx = torch.arange(0, len(box_idx), 4, dtype=torch.long, device=device)
    y_idx = torch.arange(1, len(box_idx), 4, dtype=torch.long, device=device)
    w_idx = torch.arange(2, len(box_idx), 4, dtype=torch.long, device=device)
    h_idx = torch.arange(3, len(box_idx), 4, dtype=torch.long, device=device)
    corner_y = torch.arange(num_grids, device=device)
    corner_y = corner_y.view(-1, 1, 1).repeat((1, num_grids, num_bboxes))
    corner_x = corner_y.transpose(0, 1)

    # transform
    boxes = pred_tensor[..., box_idx]
    boxes[..., x_idx] = (torch.sigmoid(boxes[..., x_idx]) + corner_x) / num_grids
    boxes[..., y_idx] = (torch.sigmoid(boxes[..., y_idx]) + corner_y) / num_grids
    boxes[..., w_idx] = (anchors[..., 0] * torch.exp(boxes[..., w_idx])) / num_grids
    boxes[..., h_idx] = (anchors[..., 1] * torch.exp(boxes[..., h_idx])) / num_grids
    confidences = torch.sigmoid(pred_tensor[..., conf_idx])
    category = pred_tensor[..., class_idx].view(num_grids, num_grids, -1, num_classes)
    category = torch.softmax(category, dim=-1)
    scores, labels = torch.max(category, dim=-1)

    # filter
    mask = confidences * scores > prob_threshold
    boxes = boxes[mask.repeat_interleave(4, dim=-1)].view(-1, 4)
    confidences = confidences[mask]
    scores = scores[mask]
    labels = labels[mask]
    return boxes, confidences, scores, labels
