from typing import Any

from torch import Tensor, nn
from torch.utils.data import DataLoader


def training_set(model: nn.Module):
    import torch

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
    return optimizer, scheduler


def cust_forward_fn(model: nn.Module, data: Any) -> Any:
    from andesPrune.utils.utils import model_device

    return model(data[0].to(device=model_device(model)))


def cust_loss_fn(output: Any, data: Any) -> Tensor:
    label = data[1].to(device=output.device)
    criterion = nn.CrossEntropyLoss()
    return criterion(output, label)


def cust_eval_fn(output: Any, data: Any) -> Tensor:
    from andesPrune.utils.topk import top1

    label = data[1].to(device=output.device)
    return top1(output, label)


def cust_inference_fn(model: nn.Module, dataset: DataLoader) -> None:
    import torch

    from andesPrune.utils.utils import average

    with torch.no_grad():
        loss, acc = [], []
        for data in dataset:
            output = cust_forward_fn(model, data)
            loss.append(cust_loss_fn(output, data).item())
            acc.append(cust_eval_fn(output, data).item())
    return {"Loss": average(loss)}, {"Top1": average(acc)}
