import torch
from torch import nn
from andesPrune.utils.topk import top1
from andesPrune.utils.utils import model_device


def training_set(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
    return optimizer, scheduler


def cust_forward_fn(model, data):
    return model(data[0].to(device=model_device(model)))


def cust_loss_fn(output, data):
    label = data[1].to(device=output.device)
    criterion = nn.CrossEntropyLoss()
    return criterion(output, label)


def distil_loss_fn(output, label):
    label = label.to(device=output.device)
    criterion = nn.KLDivLoss(reduction="batchmean")
    return criterion(output, label)


def cust_eval_fn(output, data):
    label = data[1].to(device=output.device)
    return top1(output, label)
