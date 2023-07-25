from typing import List, Union

import torch
from torch import Tensor

__all__ = ["topk", "top1", "top5"]


def topk(output: Tensor, label: Tensor, k: Union[int, list] = 1) -> Union[Tensor, List[Tensor]]:
    """Calculate Top K accuracy

    Args:
        output (Tensor): output tensor
        label (Tensor): label
        k (Union[int, list], optional): coefficient. Defaults to 1.

    Returns:
        Union[Tensor, List[Tensor]]: top k accuracy
    """
    if not type(k) in [int, list]:
        raise ValueError(f"k must be int or list, got {type(k)}")
    if not label.dtype in [
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ]:
        raise ValueError(f"label data type is signed or unsigned int, got {label.dtype}")
    if not output.dim() == 2:
        raise ValueError(f"output tensor dimension must be 2, got {output.dim()}")
    if not label.dim() == 1:
        raise ValueError(f"label tensor dimension must be 1, got {label.dim()}")
    with torch.no_grad():
        label = label.to(device=output.device).view([-1, 1])
        if type(k) is int:
            _, preds = torch.topk(output, k, dim=-1)
            results = (preds == label).sum(dim=-1)
            accuracy = (results.detach().count_nonzero() / results.detach().numel()).detach().clone()
            return accuracy
        else:
            accuracy = []
            for _k in k:
                try:
                    _, preds = torch.topk(output, _k, dim=-1)
                except RuntimeError:
                    _, preds = torch.topk(output, output.shape[-1], dim=-1)
                results = (preds == label).sum(dim=-1)
                accuracy.append((results.detach().count_nonzero() / results.detach().numel()).detach().clone())
            return accuracy


def top1(output: Tensor, label: Tensor) -> Tensor:
    """Top 1 accuracy

    Args:
        output (Tensor): output tensor
        label (Tensor): label

    Returns:
        Tensor: Top 1 accuracy
    """
    return topk(output, label, 1)


def top5(output: Tensor, label: Tensor) -> Tensor:
    """Top 5 accuracy

    Args:
        output (Tensor): output tensor
        label (Tensor): label

    Returns:
        Tensor: Top 5 accuracy
    """
    return topk(output, label, 5)
