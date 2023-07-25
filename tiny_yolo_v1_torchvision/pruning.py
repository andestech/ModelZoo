import torch
from .voc import VOC_CLASS
from .utils import decode_v1, detect, evaluate_map_voc
from .model_fp32 import return_fp32_model

loss = return_fp32_model().loss


def training_set(model):
    optimizer = torch.optim.SGD(model.parameters(), 1e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    return optimizer, scheduler


def cust_forward_fn(model, data):
    device = next(model.parameters()).device
    return model(data[0].to(device=device))


def cust_loss_fn(output, data):
    label = data[1].to(device=output.device)
    return loss(output, label)


def cust_inference_fn(model, dataloader):
    device = next(model.parameters()).device
    model.eval()
    decode = decode_v1
    with torch.no_grad():
        a, b, c, d, e, f, g, h = [], [], [], [], [], [], [], []
        for id in range(len(dataloader.dataset)):
            id = torch.tensor(id).to(device)
            input, label = dataloader.dataset[id]
            boxes_detected, classes_detected, probs_detected = detect(
                input.to(device), model, decode, 0.5, num_classes=20, prob_threshold=0.01
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
    return {}, {"mAP": aps.mean().item()}
