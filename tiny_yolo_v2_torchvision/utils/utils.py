from typing import Callable, Tuple

from torch import LongTensor, Tensor, cat, max, min, nn, no_grad, zeros

__all__ = ["compute_iou", "nms", "detect", "detect_tflite"]


def compute_iou(
    bbox1: Tensor, bbox2: Tensor, compare_all: bool = False, x1y1x2y2: bool = False
) -> Tensor:
    """
    Compute the IoU (Intersection over Union) of two set of bboxes.
    (bbox format: if x1y1x2y2 is True; the coordinate is [x1, y1, x2, y2], else [x_c, y_c, w, h])
    Args:
        bbox1: (Tensor) bounding bboxes, sized [(B), N, 4].
        bbox2: (Tensor) bounding bboxes, sized [(B), M, 4].
        compare_all (bool): calculate ious of every bbox1 to all bbox2
        x1y1x2y2 (bool): if using x1y1x2y2 coordination set True, cxcywh set False
    Returns:
        IoU: (Tensor) if compare_all is True which size is [(B), N, M], else size is [(B), N].
    """
    assert bbox1.size(-1) == 4, f"bbox1 coordinate representation is 4, got {bbox1.size(-1)}"
    assert bbox2.size(-1) == 4, f"bbox2 coordinate representation is 4, got {bbox2.size(-1)}"
    device = bbox1.device
    if not x1y1x2y2:
        new_bbox1 = zeros(bbox1.shape, device=device)
        new_bbox1[..., 0:2] = bbox1[..., :2] - 0.5 * bbox1[..., 2:4]
        new_bbox1[..., 2:4] = bbox1[..., :2] + 0.5 * bbox1[..., 2:4]

        new_bbox2 = zeros(bbox2.shape, device=device)
        new_bbox2[..., 0:2] = bbox2[..., :2] - 0.5 * bbox2[..., 2:4]
        new_bbox2[..., 2:4] = bbox2[..., :2] + 0.5 * bbox2[..., 2:4]
    else:
        new_bbox1 = bbox1.clone()
        new_bbox2 = bbox2.clone()

    if compare_all:
        if new_bbox1.dim() == 2 and bbox2.dim() == 2:
            B = 1  # batch_size
            N = new_bbox1.size(0)
            M = new_bbox2.size(0)
            new_bbox1 = new_bbox1.unsqueeze(0)
            new_bbox2 = new_bbox2.unsqueeze(0)
        elif (
            new_bbox1.dim() == 3 and new_bbox2.dim() == 3 and new_bbox1.size(0) == new_bbox2.size(0)
        ):
            B = new_bbox1.size(0)  # batch_size
            N = new_bbox1.size(1)
            M = new_bbox2.size(1)
        else:
            raise ValueError(
                f"Compare_all argument only support bounding boxes with 2 or 3 dimensions."
            )
        # Compute left-top coordinate of the intersections
        lt = max(
            new_bbox1[..., 0:2].unsqueeze(2).expand(B, N, M, 2),  # [B, N, 2] -> [B, N, 1, 2] -> [B, N, M, 2]
            new_bbox2[..., 0:2].unsqueeze(1).expand(B, N, M, 2),  # [B, M, 2] -> [B, 1, M, 2] -> [B, N, M, 2]
        )

        # Conpute right-bottom coordinate of the intersections
        rb = min(
            new_bbox1[..., 2:4].unsqueeze(2).expand(B, N, M, 2),
            new_bbox2[..., 2:4].unsqueeze(1).expand(B, N, M, 2),
        )

        # Compute area of the intersections from the coordinates
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]  # [B, N, M]

        # Compute area of the bboxes
        area1 = (new_bbox1[..., 2:4] - new_bbox1[..., :2]).clamp(min=0)  # [B, N]
        area2 = (new_bbox2[..., 2:4] - new_bbox2[..., :2]).clamp(min=0)
        area1 = area1[..., 0] * area1[..., 1]
        area2 = area2[..., 0] * area2[..., 1]
        area1 = area1.unsqueeze(2).expand_as(inter)  # [B, N] -> [B, N, 1] -> [B, N, M]
        area2 = area2.unsqueeze(1).expand_as(inter)  # [B, M] -> [B, 1, M] -> [B, N, M]
    else:
        assert (
            new_bbox1.shape == new_bbox2.shape
        ), f"bboxes1 has same shape as bboxes2, got bbox1[{new_bbox1.shape}] and bbox2[{new_bbox2.shape}]"
        # Compute left-top coordinate of the intersections, size [B, N, 2]
        lt = max(new_bbox1[..., 0:2], new_bbox2[..., 0:2])

        # Conpute right-bottom coordinate of the intersections, size [B, N, 2]
        rb = min(new_bbox1[..., 2:4], new_bbox2[..., 2:4])

        # Compute area of the intersections from the coordinates
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]  # [B, N]

        # Compute area of the bboxes
        area1 = (new_bbox1[..., 2:4] - new_bbox1[..., :2]).clamp(min=0)  # [B, N]
        area2 = (new_bbox2[..., 2:4] - new_bbox2[..., :2]).clamp(min=0)
        area1 = area1[..., 0] * area1[..., 1]
        area2 = area2[..., 0] * area2[..., 1]

    # Compute IoU from the areas
    union = area1 + area2 - inter
    iou = (inter / (union + 1e-6)).squeeze(0)
    return iou


def nms(
    boxes: Tensor, confidences: Tensor, nms_threshold: float = 0.5, x1y1x2y2: bool = False
) -> LongTensor:
    """Apply non maximum supression.
    Args:
        boxes (tensor): [[x_c, y_c, w, h], ...]. Normalized from 0.0 to 1.0 w.r.t. image width/height, sized [n_boxes, 4].
        confidences (tensor): objectness confidences for each detected box, sized [n_boxes].
        nms_threshold (float): also iou threshold
    Returns:
        idx (tensor): remained bounding boxes indices
    """
    assert isinstance(boxes, Tensor), f"Expected boxes type is {Tensor}, but got {type(boxes)}"
    assert isinstance(
        confidences, Tensor
    ), f"Expected boxes type is {Tensor}, but got {type(confidences)}"
    assert (
        boxes.dim() == 2 and boxes.size(-1) == 4
    ), f"Expected boxes second dimension is [n_boxes, 4], got {boxes.shape}"
    assert (
        confidences.dim() == 1
    ), f"Expected boxes second dimension is [n_boxes], got {confidences.shape}"

    _, idx_sorted = confidences.sort(descending=True)
    keep_idx = []
    while idx_sorted.numel() > 0:
        idx = idx_sorted[0]
        keep_idx.append(idx)
        idx_sorted = idx_sorted[1:]
        ious = compute_iou(boxes[idx].unsqueeze(0), boxes[idx_sorted], compare_all=True, x1y1x2y2=x1y1x2y2)[0]
        keep_ids = (ious <= nms_threshold).nonzero().view(-1)
        idx_sorted = idx_sorted[keep_ids]
        if idx_sorted.numel() == 1:
            keep_idx.append(idx_sorted)
            break
    return LongTensor(keep_idx)


def detect(
    image: Tensor,
    model: nn.Module,
    decode: Callable[[Tensor], Tuple[Tensor]],
    nms_threshold: float = 0.5,
    **kwargs
):
    """Detect objects from given image.
    Args:
        image: (Tensor) input image in RGB, sized [3, h, w].
    Returns:
        boxes_detected (Tensor): normalized boxes corner, width and height, size [n_bboxes, 4].
        classes_detected (Tensor): detected classes with one-hot encoding, size [n_bboxes, n_classes].
        probs_detected (Tensor): probability(=confidence x class_score) for each detected box, size [n_bboxes, 1].
    """
    if image.dim() == 4:
        image = image[0]
    with no_grad():
        model.eval()
        pred_tensor = model(image.unsqueeze(0))
        if isinstance(pred_tensor, list):
            pred_tensor = [x.squeeze(0) for x in pred_tensor]
        else:
            pred_tensor = pred_tensor.squeeze(0)
    # Get detected boxes_detected, labels, confidences, class-scores.
    boxes, confidences, scores, labels = decode(pred_tensor, **kwargs)
    if len(boxes) == 0:
        return Tensor(), Tensor(), Tensor()
    # Apply non maximum supression for boxes of each class.
    boxes_detected, classes_detected, probs_detected = [], [], []
    for class_idx in range(kwargs["num_classes"]):
        mask = labels == class_idx
        boxes_masked = boxes[mask]
        confidences_masked = confidences[mask]
        labels_maked = labels[mask]
        scores_masked = scores[mask]

        if len(boxes_masked) == 0:
            continue
        idx = nms(boxes_masked, confidences_masked, nms_threshold)
        boxes_detected.append(boxes_masked[idx])
        classes_detected.append(labels_maked[idx])
        probs_detected.append(confidences_masked[idx] * scores_masked[idx])

    boxes_detected = cat(boxes_detected)
    classes_detected = cat(classes_detected)
    probs_detected = cat(probs_detected)
    return boxes_detected, classes_detected, probs_detected

def detect_tflite(
    pred_tensor,
    decode: Callable[[Tensor], Tuple[Tensor]],
    nms_threshold: float = 0.5,
    **kwargs
):
    """Detect objects from given image.
    Args:
        image: (Tensor) input image in RGB, sized [3, h, w].
    Returns:
        boxes_detected (Tensor): normalized boxes corner, width and height, size [n_bboxes, 4].
        classes_detected (Tensor): detected classes with one-hot encoding, size [n_bboxes, n_classes].
        probs_detected (Tensor): probability(=confidence x class_score) for each detected box, size [n_bboxes, 1].
    """

    if isinstance(pred_tensor, list):
        pred_tensor = [x.squeeze(0) for x in pred_tensor]
    else:
        pred_tensor = pred_tensor.squeeze(0)
    # Get detected boxes_detected, labels, confidences, class-scores.
    boxes, confidences, scores, labels = decode(pred_tensor, **kwargs)
    if len(boxes) == 0:
        return Tensor(), Tensor(), Tensor()

    # Apply non maximum supression for boxes of each class.
    boxes_detected, classes_detected, probs_detected = [], [], []
    for class_idx in range(kwargs["num_classes"]):
        mask = labels == class_idx
        boxes_masked = boxes[mask]
        confidences_masked = confidences[mask]
        labels_maked = labels[mask]
        scores_masked = scores[mask]

        if len(boxes_masked) == 0:
            continue
        idx = nms(boxes_masked, confidences_masked, nms_threshold)
        boxes_detected.append(boxes_masked[idx])
        classes_detected.append(labels_maked[idx])
        probs_detected.append(confidences_masked[idx] * scores_masked[idx])

    boxes_detected = cat(boxes_detected)
    classes_detected = cat(classes_detected)
    probs_detected = cat(probs_detected)
    return boxes_detected, classes_detected, probs_detected

if __name__ == "__main__":
    a = zeros((1, 1, 4))
    a[0][0][0] = 0
    a[0][0][1] = 0
    a[0][0][2] = 6
    a[0][0][3] = 6

    b = zeros((1, 1, 4))
    b[0][0][0] = 1
    b[0][0][1] = 1
    b[0][0][2] = 4
    b[0][0][3] = 4

    iou1 = compute_iou(a.squeeze(0), b.squeeze(0))
    print(iou1)
