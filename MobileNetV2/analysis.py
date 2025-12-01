import os
import yaml
import torch
from tqdm import tqdm
from andesQuant.quantize_utils.analysis_utils import Analysis_block

now_dir = os.path.dirname(__file__)
with open(now_dir + "/model_cfg.yaml", 'r') as f:
    input_yaml = yaml.load(f, Loader=yaml.FullLoader)


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
            input.div_(scale).sub_(zero_point).sub_(128).round_().clamp_(-128, 127)
    else:
        input.div_(scale).sub_(zero_point).round_().clamp_(clamp_min, clamp_max).add_(zero_point).mul_(scale)


def output_generate(model, dataloader, device):
    model.eval()
    device = "cpu"
    model.to(device)
    print("Output generating")
    count = 0

    with torch.no_grad():
        for image, label in tqdm(dataloader):
            image.to(device)
            label.to(device)
            logits = model(image)
            torch.save(logits, './analysis_tmp/target_' + str(count) + '.pt')
            if count == 20:
                return 0.0
            count = count + 1


def sensitivity_trace(model, dataloaders, device, grad_trace=False):
    model.eval()
    device = "cpu"
    model.to(device)
    print("Start sensitivity trace")
    count = 0
    total_loss = 0
    criterion = torch.nn.MSELoss()

    if grad_trace:
        total_count = 0 # Avoid gradients to exceed memory
        for name, module in model.named_modules():
            if isinstance(module, Analysis_block):
                module.grad_trace = True
    else:
        total_count = 10 # For accumulating more SQNR       

    for inputs, labels in tqdm(dataloaders):
        inputs = inputs.to(device)
        labels = labels.to(device)
        if grad_trace:
            inputs.requires_grad = True
        outputs = model(inputs)
        if grad_trace:
            target = torch.load('./analysis_tmp/target_' + str(count) + '.pt').to(device)
            loss = criterion(outputs, target)
            total_loss=loss + total_loss
        if count == total_count:
            if grad_trace:
                total_loss.backward(create_graph=True)
            return model
        count = count + 1
