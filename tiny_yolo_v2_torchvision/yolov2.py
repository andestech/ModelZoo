import torch
import torch.nn.functional as F
from ModelZoo.tiny_yolo_v2_torchvision.utils import (
    compute_iou, 
    detect,
    YOLOv2TrainPostProcessing,
    create_network,
    decode_v2,
    forward,
    load_weights,
    parse_cfg,
    save_weights,
    evaluate_map_voc,
)
from ModelZoo.tiny_yolo_v2_torchvision.dataset import VOC_CLASS, voc_detection
from torch import Tensor, nn, ones_like, zeros_like

__all__ = ["Tiny_YOLOv2"]

# support route shortcut and reorg
class Tiny_YOLOv2(nn.Module):
    def __init__(
        self,
        cfgfile: str,
        num_grids: int = 13,
        lambda_coord: float = 1.0,
        lambda_respon: float = 5.0,
        lambda_noobj: float = 1.0,
        lambda_class: float = 1.0,
        prob_threshold: float = 0.01,
        nms_threshold: float = 0.5,
    ):
        super(Tiny_YOLOv2, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = create_network(self, self.blocks)
        self.num_grids = num_grids
        self.lambda_coord = lambda_coord
        self.lambda_respon = lambda_respon
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        self.prob_threshold = prob_threshold
        self.nms_threshold = nms_threshold

    def forward(self, x: Tensor):
        return forward(self.blocks, self.models, x)

    def loss(self, pred_tensor: Tensor, target_tensor: Tensor):
        """Compute loss for YOLO training.
        Args:
            pred_tensor: (Tensor) predictions, sized [n_batches, S, S, B x (5 + C)], 5=len([x, y, w, h, conf]).
            target_tensor: (Tensor) targets, sized [n_batches, S, S, B x (5 + C)].
        Returns:
            loss (Tensor)
        """
        device = pred_tensor.device
        step = int(5 + self.num_classes)
        num_bboxes = int(pred_tensor.size(-1) // step)
        num_grids = int(pred_tensor.size(1))
        anchors = torch.tensor(self.anchors, dtype=pred_tensor.dtype, device=device)

        # idx
        box_idx = torch.cat([torch.arange(i * step, i * step + 4) for i in range(num_bboxes)]).to(device)
        conf_idx = torch.arange(4, num_bboxes * step, step).to(device)
        class_idx = torch.cat(
            [torch.arange(5 + i * step, 5 + i * step + self.num_classes) for i in range(num_bboxes)]
        ).to(device)
        x_idx = torch.arange(0, len(box_idx), 4, dtype=torch.long, device=device)
        y_idx = torch.arange(1, len(box_idx), 4, dtype=torch.long, device=device)
        w_idx = torch.arange(2, len(box_idx), 4, dtype=torch.long, device=device)
        h_idx = torch.arange(3, len(box_idx), 4, dtype=torch.long, device=device)
        corner_y = torch.arange(num_grids, device=device)
        corner_y = corner_y.view(-1, 1, 1).repeat((1, num_grids, num_bboxes))
        corner_x = corner_y.transpose(0, 1)

        # [n_batches, S, S, B, 4xB]
        pre_boxes = pred_tensor[..., box_idx]
        target_boxes = target_tensor[..., box_idx]
        # [n_batches, S, S, B]
        pre_confidences = pred_tensor[..., conf_idx]
        target_confidences = target_tensor[..., conf_idx]
        # [n_batches, S, S, B, C]
        pre_class_scores = pred_tensor[..., class_idx].view(-1, num_grids, num_grids, num_bboxes, self.num_classes)
        target_class_scores = target_tensor[..., class_idx].view(-1, num_grids, num_grids, num_bboxes, self.num_classes)

        # decode
        pre_boxes[..., x_idx] = torch.sigmoid(pre_boxes[..., x_idx])
        pre_boxes[..., y_idx] = torch.sigmoid(pre_boxes[..., y_idx])
        pre_confidences = torch.sigmoid(pre_confidences)

        # ious
        _pre_boxes = pre_boxes.clone()
        _target_boxes = target_boxes.clone()
        _pre_boxes[..., x_idx] = (_pre_boxes[..., x_idx] + corner_x) / num_grids
        _pre_boxes[..., y_idx] = (_pre_boxes[..., y_idx] + corner_y) / num_grids
        _pre_boxes[..., w_idx] = anchors[..., 0] * torch.exp(_pre_boxes[..., w_idx]) / num_grids
        _pre_boxes[..., h_idx] = anchors[..., 1] * torch.exp(_pre_boxes[..., h_idx]) / num_grids
        _target_boxes[..., x_idx] = (_target_boxes[..., x_idx] + corner_x) / num_grids
        _target_boxes[..., y_idx] = (_target_boxes[..., y_idx] + corner_y) / num_grids
        _target_boxes[..., w_idx] = anchors[..., 0] * torch.exp(_target_boxes[..., w_idx]) / num_grids
        _target_boxes[..., h_idx] = anchors[..., 1] * torch.exp(_target_boxes[..., h_idx]) / num_grids
        ious = compute_iou(
            _pre_boxes.view(-1, num_grids, num_grids, num_bboxes, 4),
            _target_boxes.view(-1, num_grids, num_grids, num_bboxes, 4),
        )
        _pre_boxes[..., x_idx] = 0
        _pre_boxes[..., y_idx] = 0
        _pre_boxes[..., w_idx] = anchors[..., 0] / num_grids
        _pre_boxes[..., h_idx] = anchors[..., 1] / num_grids
        _target_boxes[..., x_idx] = 0
        _target_boxes[..., y_idx] = 0
        ious_mask = compute_iou(
            _pre_boxes.view(-1, num_grids, num_grids, num_bboxes, 4),
            _target_boxes.view(-1, num_grids, num_grids, num_bboxes, 4),
        ).detach()

        ious *= target_confidences
        ious_mask *= target_confidences
        ious_max = ious_mask.amax(dim=-1, keepdim=True)
        respon_mask = (ious_mask == ious_max) & (ious_max != 0)
        nonrespon_mask = (ious_mask < self.ignore_threshold) & ~respon_mask

        # ratio
        ratio = 2 - target_boxes[..., w_idx][respon_mask] * target_boxes[..., h_idx][respon_mask]

        # 1. truth coordinate loss
        loss_x = F.mse_loss(pre_boxes[..., x_idx][respon_mask], target_boxes[..., x_idx][respon_mask], reduction="none")
        loss_y = F.mse_loss(pre_boxes[..., y_idx][respon_mask], target_boxes[..., y_idx][respon_mask], reduction="none")
        loss_w = F.mse_loss(pre_boxes[..., w_idx][respon_mask], target_boxes[..., w_idx][respon_mask], reduction="none")
        loss_h = F.mse_loss(pre_boxes[..., h_idx][respon_mask], target_boxes[..., h_idx][respon_mask], reduction="none")

        # 2. object confidence loss:
        obj_conf_loss = F.mse_loss(pre_confidences[respon_mask], ious[respon_mask].detach(), reduction="sum")

        # 3. no object confidence loss
        nonrespon_conf_loss = F.mse_loss(
            pre_confidences[nonrespon_mask], zeros_like(pre_confidences[nonrespon_mask]), reduction="sum"
        )

        # 4. class loss
        class_loss = F.mse_loss(
            pre_class_scores.softmax(dim=-1)[respon_mask], target_class_scores[respon_mask], reduction="sum"
        )

        # 5. iou loss
        iou_loss = F.mse_loss(ious[respon_mask], ones_like(ious[respon_mask]).detach(), reduction="sum")

        loss = (
            self.lambda_coord * (ratio * (loss_x + loss_y + loss_w + loss_h)).sum()
            + self.lambda_respon * obj_conf_loss
            + self.lambda_noobj * nonrespon_conf_loss
            + self.lambda_class * class_loss
            + iou_loss
        )
        return loss.sum() / pred_tensor.size(0)


if __name__ == "__main__":
    device = "cuda:0"
    model = Tiny_YOLOv2("ModelZoo/tiny_yolo_v2_torchvision/yolov2.cfg").to(device)
    load_weights(model, "ModelZoo/tiny_yolo_v2_torchvision/yolov2-voc.weights", version="v3")
    optim = torch.optim.SGD(model.parameters(), 0.0001, momentum=0.9, weight_decay=0.0005)
    # sched = torch.optim.lr_scheduler.CyclicLR(
    #     optim, base_lr=0.00001, max_lr=0.001, base_momentum=0.1, max_momentum=0.9, mode="triangular2", step_size_up=10
    # )
    dataloader = voc_detection(
        img_size=(model.input_width, model.input_height),
        postprocessing=YOLOv2TrainPostProcessing(model.num_grids, model.num_bboxes, model.num_classes, model.anchors),
        batch_size=64,
    )

    model = nn.parallel.DataParallel(model, [0, 1])
    best = 0
    for epoch in range(250):
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

        model.eval()
        a, b, c, d, e, f, g, h = [], [], [], [], [], [], [], []
        for id in range(len(dataloader["test"].dataset)):
            id = torch.tensor(id).to(device)
            input, label = dataloader["test"].dataset[id]
            boxes_detected, classes_detected, probs_detected = detect(
                input.to(device),
                model,
                decode_v2,
                model.module.nms_threshold,
                anchors=model.module.anchors,
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
            # plot_boxes(
            #     input.permute(1, 2, 0).cpu().numpy(),
            #     boxes_detected.cpu().numpy(),
            #     classes_detected.cpu().numpy(),
            #     class_names=VOC_CLASS,
            # )
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
                torch.save(model.state_dict(), "rdca_ai_pattern/weights/yolov2.pt")
