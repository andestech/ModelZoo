import torch
import numpy as np
from tqdm import tqdm
from andesQuant.quantize_utils.analysis_utils import Analysis_block


"""
Pre_qunat model input tools
"""
def QuantStub(input, min_val=-1.0, max_val=1.0, symm=True, bits=8, isHW=False):
    assert max_val > min_val, "max_val must larger than min_val"
    if symm:
        clamp_min = -((2 ** (bits - 1)))
        clamp_max = 2 ** (bits - 1) - 1
        scale = torch.max(torch.tensor(min_val).abs(), torch.tensor(max_val).abs()).div((2 ** (bits - 1)) - 1)
        zero_point = torch.tensor(0.0)
    else:
        clamp_min = 0
        clamp_max = (2 ** bits) - 1
        scale = (torch.tensor(max_val) - torch.tensor(min_val)).div((2 ** bits) - 1)
        zero_point = torch.tensor(min_val).div(scale).round()
    if isHW:
        if symm:
            input.div_(scale).sub_(zero_point).round_().clamp_(clamp_min, clamp_max)
        else:
            input.div_(scale).sub_(zero_point).sub_(128).round_().clamp_(-128, 127).add_(128).add_(zero_point)
    else:
        input.div_(scale).sub_(zero_point).round_().clamp_(clamp_min, clamp_max).add_(zero_point).mul_(scale)


def output_generate(model, dataloader, device):
    model.eval()
    device = "cpu"
    model.to(device)
    print("Output generating")
    count = 0

    with torch.no_grad():
        for ii, sample in enumerate(dataloader):
            image, label = sample[0].to(device), sample[1].numpy()
            logits = model(image)
            torch.save(logits, './analysis_tmp/target_' + str(count) + '.pt')
            if count==20:
                return 0.0
            count = count+1


def sensitivity_trace(model, dataloaders, device, grad_trace=False):
    model.eval()
    device = "cpu"
    model.to(device)
    print("Start sensitivity trace")
    count=0
    total_loss = 0
    criterion = torch.nn.MSELoss()

    if grad_trace:
        total_count = 0 # Avoid gradients to exceed memory
        for name, module in model.named_modules():
            if isinstance(module, Analysis_block):
                module.grad_trace = True
    else:
        total_count = 10 # For accumulating more SQNR

    for ii, sample in enumerate(dataloaders):
        image, label = sample[0].to(device), sample[1].numpy()
        if grad_trace:
            image.requires_grad = True
        logits = model(image)
        if grad_trace:
            target = torch.load('./analysis_tmp/target_' + str(count) + '.pt').to(device)
            loss = criterion(logits, target)
        if count == total_count:
            if grad_trace:
                loss.backward(create_graph=True)
            return model
        count = count + 1
