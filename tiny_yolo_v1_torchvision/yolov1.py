import numpy as np
import torch
from ModelZoo.tiny_yolo_v1_torchvision.utils import (
    YOLOv1TrainPostProcessing,
    create_network,
    decode_v1,
    forward,
    parse_cfg,
    evaluate_map_voc,
    load_weights,
    compute_iou,
    detect,
)
from ModelZoo.tiny_yolo_v1_torchvision.voc import VOC_CLASS, voc_detection
from torch import Tensor, nn, zeros_like
from torch.nn import functional as F

__all__ = ["Tiny_YOLOv1"]


class Tiny_YOLOv1(nn.Module):
    def __init__(
        self,
        cfgfile: str,
        num_grids: int = 7,
        lambda_coord: float = 5.0,
        lambda_respon: float = 1.0,
        lambda_noobj: float = 0.5,
        lambda_class: float = 1.0,
        prob_threshold: float = 0.01,
        nms_threshold: float = 0.5,
    ):
        super().__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = create_network(self, self.blocks)
        self.num_grids = num_grids
        self.lambda_coord = lambda_coord
        self.lambda_respon = lambda_respon
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        self.out_dim = int(5 * self.num_bboxes + self.num_classes)
        self.num_grids = num_grids
        self.prob_threshold = prob_threshold
        self.nms_threshold = nms_threshold

    def forward(self, x: Tensor):
        return (
            forward(self.blocks, self.models, x)
            .view(-1, self.out_dim, self.num_grids, self.num_grids)
            .permute(0, 2, 3, 1)
        )

    def loss(self, pred_tensor: Tensor, target_tensor: Tensor):
        """Compute loss for YOLO training.
        Args:
            pred_tensor: (Tensor) predictions, sized [n_batches, S, S, 5 x B + C], 5=len([x, y, w, h, conf]).
            target_tensor: (Tensor) targets, sized [n_batches, S, S, 5 x B + C].
        Returns:
            loss (Tensor)
        """
        device = pred_tensor.device
        num_bboxes: int = self.num_bboxes

        # Since there is the indicator function in the Yolo's objective function, we first attain the mask with the following code.
        conf_idx = torch.arange(4, 5 * num_bboxes, 5).long()
        box_idx = np.arange(5 * num_bboxes)
        box_idx = torch.tensor(np.delete(box_idx, conf_idx), device=device).long()
        conf_idx = conf_idx.to(device)
        x_idx = torch.arange(0, len(box_idx), 4, device=device)
        y_idx = torch.arange(1, len(box_idx), 4, device=device)
        w_idx = torch.arange(2, len(box_idx), 4, device=device)
        h_idx = torch.arange(3, len(box_idx), 4, device=device)
        corner_y = torch.arange(self.num_grids, dtype=pred_tensor.dtype, device=device)
        corner_y = corner_y.view(-1, 1, 1).expand((self.num_grids, self.num_grids, num_bboxes))
        corner_x = corner_y.transpose(0, 1)

        # [n_batches, S, S, B, 4xB]
        pre_boxes = pred_tensor[..., box_idx]
        target_boxes = target_tensor[..., box_idx]
        # [n_batches, S, S, B]
        pre_confidences = pred_tensor[..., conf_idx]
        target_confidences = target_tensor[..., conf_idx]
        # [n_batches, S, S, C]
        pre_class_scores = pred_tensor[..., 5 * num_bboxes :]
        target_class_scores = target_tensor[..., 5 * num_bboxes :]

        # ious
        _pre_boxes = pre_boxes.detach().clone()
        _target_boxes = target_boxes.detach().clone()
        _pre_boxes[..., x_idx] = (_pre_boxes[..., x_idx] + corner_x) / self.num_grids
        _pre_boxes[..., y_idx] = (_pre_boxes[..., y_idx] + corner_y) / self.num_grids
        _target_boxes[..., x_idx] = (_target_boxes[..., x_idx] + corner_x) / self.num_grids
        _target_boxes[..., y_idx] = (_target_boxes[..., y_idx] + corner_y) / self.num_grids
        ious = compute_iou(
            _pre_boxes.view(-1, self.num_grids, self.num_grids, num_bboxes, 4),
            _target_boxes.view(-1, self.num_grids, self.num_grids, num_bboxes, 4),
        )

        # mask
        obj_mask = target_confidences.bool()
        ious = ious * obj_mask  # masked non-object ious
        respon_mask = (ious > 0) & (ious == ious.amax(dim=-1, keepdim=True))  # keep max
        nonrespon_mask = ~respon_mask | ~obj_mask

        # Notice: must pay attention to gradient, if pre_boxes using sqrt will occur gradient explosion)
        # 1. coordinate loss
        loss_x = F.mse_loss(pre_boxes[..., x_idx][respon_mask], target_boxes[..., x_idx][respon_mask], reduction="sum")
        loss_y = F.mse_loss(pre_boxes[..., y_idx][respon_mask], target_boxes[..., y_idx][respon_mask], reduction="sum")
        loss_w = F.mse_loss(pre_boxes[..., w_idx][respon_mask], target_boxes[..., w_idx][respon_mask], reduction="sum")
        loss_h = F.mse_loss(pre_boxes[..., h_idx][respon_mask], target_boxes[..., h_idx][respon_mask], reduction="sum")

        # 2. response confidence loss:
        obj_conf_loss = F.mse_loss(pre_confidences[respon_mask], ious[respon_mask], reduction="sum")

        # 3. non-response confidence loss
        nonrespon_conf_loss = F.mse_loss(
            pre_confidences[nonrespon_mask], zeros_like(pre_confidences[nonrespon_mask]), reduction="sum"
        )

        # 4. class loss
        obj_mask = obj_mask.sum(dim=-1, keepdim=True).bool().repeat_interleave(self.num_classes, dim=-1)
        class_loss = F.mse_loss(pre_class_scores[obj_mask], target_class_scores[obj_mask], reduction="sum")

        loss = (
            self.lambda_coord * (loss_x + loss_y + loss_w + loss_h)
            + self.lambda_respon * obj_conf_loss
            + self.lambda_noobj * nonrespon_conf_loss
            + self.lambda_class * class_loss
        )
        return loss / pred_tensor.size(0)


def train_object_detection():
    device = "cuda:0"
    model = Tiny_YOLOv1("ModelZoo/tiny_yolo_v1_torchvision/yolov1-tiny.cfg").to(device)
    optim = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, weight_decay=5e-4)
    # sched = torch.optim.lr_scheduler.CyclicLR(
    #     optim, base_lr=0.00001, max_lr=0.0001, base_momentum=0.3, mode="triangular2", step_size_up=3000
    # )
    # load_weights(model, "weights/tiny-yolov1.weights", "v1")
    model.load_state_dict(
        torch.load("ModelZoo/tiny_yolo_v1_torchvision/weights/yolov1.pt", map_location=device)
    )
    dataloader = voc_detection(
        img_size=(448, 448),
        postprocessing=YOLOv1TrainPostProcessing(model.num_grids, model.num_bboxes, model.num_classes),
        batch_size=128,
    )

    best = 0
    model = nn.parallel.DataParallel(model, [0, 1])
    for epoch in range(1000):
        # training
        model.train()
        total_loss = []
        for data in dataloader["train"]:
            optim.zero_grad()
            input = data[0].to(device)
            label = data[1].to(device)
            output = model(input)
            loss = model.module.loss(output, label)
            loss.backward()
            optim.step()
            # sched.step()
            total_loss.append(loss.item())
        total_loss = sum(total_loss) / len(total_loss)
        print(f"Training - Epoch: {epoch:4} | Loss: {total_loss:.5f}")
        if epoch == 0:
            optim.param_groups[0]["lr"] = 0.00125
        if epoch == 75:
            optim.param_groups[0]["lr"] = 0.001
        if epoch == 135:
            optim.param_groups[0]["lr"] = 0.00005
        if epoch == 500:
            optim.param_groups[0]["lr"] = 0.00001

        # inference
        model.eval()
        a, b, c, d, e, f, g, h = [], [], [], [], [], [], [], []
        for id in range(len(dataloader["test"].dataset)):
            id = torch.tensor(id).to(device)
            input, label = dataloader["test"].dataset[id]
            boxes_detected, classes_detected, probs_detected = detect(
                input.to(device),
                model,
                decode_v1,
                model.module.nms_threshold,
                num_classes=model.module.num_classes,
                prob_threshold=model.module.prob_threshold,
            )
            target_bboxes = label[0].to(device)
            target_classes = label[1].to(device)
            target_difficults = label[2].to(device)

            for box_detected, class_detected, prob_detected in zip(boxes_detected, classes_detected, probs_detected):
                a.append(box_detected)
                b.append(class_detected)
                c.append(prob_detected)
                d.append(id)
            for target_bbox, target_class, target_difficult in zip(target_bboxes, target_classes, target_difficults):
                e.append(target_bbox)
                f.append(target_class)
                g.append(target_difficult)
                h.append(id)

        if len(a) != 0:
            aps = evaluate_map_voc(
                torch.stack(a),
                torch.tensor(b),
                torch.tensor(c),
                torch.tensor(d),
                torch.stack(e),
                torch.tensor(f),
                torch.tensor(g),
                torch.tensor(h),
                VOC_CLASS,
            )
            for idx, ap in enumerate(aps):
                print(f"{VOC_CLASS[idx]:15s} | AP: {ap:.2%}")
            mAP = aps.mean()
            print(f"---mAP {mAP:.2%}---")
            if mAP > best:
                best = mAP
                torch.save(model.module.state_dict(), "weights/yolov1.pt")


if __name__ == "__main__":
    train_object_detection()
