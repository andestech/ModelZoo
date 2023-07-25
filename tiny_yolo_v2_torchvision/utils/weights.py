import numpy as np
import torch
from torch import nn

__all__ = ["save_weights", "load_weights"]


def save_weights(
    module_list,
    blocks,
    savedfile,
    cutoff: int = 0,
    header: list = None,
    seen: int = 0,
    version: str = "v1",
):
    if cutoff <= 0:
        cutoff = len(blocks) - 1

    with open(savedfile, "wb") as fp:
        # Attach the header at the top of the file
        if header == None:
            if version == "v1" or version == "v2":
                header = np.array([0, 0, 0, 0]).tofile(fp)
            elif version == "v3":
                header = np.array([0, 0, 0, 0, 0]).tofile(fp)
        header[3] = seen
        header = header.numpy()
        header.tofile(fp)

        # Now, let us save the weights
        for i in range(len(module_list)):
            if blocks[i + 1]["type"] == "convolutional":
                model = module_list[i]
                try:
                    batch_normalize = int(blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv: nn.Conv2d = model[0]

                if batch_normalize:
                    bn: nn.BatchNorm2d = model[1]
                    bn.bias.data.clone().cpu().numpy().tofile(fp)
                    bn.weight.data.clone().cpu().numpy().tofile(fp)
                    bn.running_mean.clone().cpu().numpy().tofile(fp)
                    bn.running_var.clone().cpu().numpy().tofile(fp)

                if conv.bias != None:
                    conv.bias.data.clone().cpu().numpy().tofile(fp)

                # Let us save the weights for the Convolutional layers
                conv.weight.data.clone().cpu().numpy().tofile(fp)

            elif blocks[i + 1]["type"] == "connected":
                model = module_list[i]
                if blocks[i + 1]["activation"] != "linear":
                    fc_model: nn.Linear = model
                else:
                    fc_model: nn.Linear = model[0]
                fc_model.bias.data.clone().cpu().numpy().tofile(fp)
                fc_model.weight.data.clone().cpu().numpy().tofile(fp)


def load_weights(model: nn.Module, weights_path: str, version: str):
    """Parses and loads the weights stored in 'weights_path'"""
    # Open the weights file
    with open(weights_path, "rb") as f:
        # ***
        # For yolo v3, the header count = 5, while it will be 4 in v1 and v2 version.
        # ***
        if version == "v1" or version == "v2":
            header = np.fromfile(f, dtype=np.int32, count=4)  # First five are header values
        elif version == "v3":
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
        header_info = header  # Needed to write header when saving weights
        seen = header[3]  # number of images seen during training
        weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

    ptr = 0
    conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv = module

        elif isinstance(module, nn.BatchNorm2d):
            # Load BN bias, weights, running mean and running variance
            bn_layer = module
            num_b = bn_layer.bias.numel()  # Number of biases
            # Bias
            bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
            bn_layer.bias.data.copy_(bn_b)
            ptr += num_b

            # Weight
            bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
            bn_layer.weight.data.copy_(bn_w)
            ptr += num_b

            # Running Mean
            bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
            bn_layer.running_mean.data.copy_(bn_rm)
            ptr += num_b

            # Running Var
            bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
            bn_layer.running_var.data.copy_(bn_rv)
            ptr += num_b

            # Convolution
            if conv is not None:
                if conv.bias is not None:
                    num_b = conv.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv.bias)
                    conv.bias.data.copy_(conv_b)
                    ptr += num_b
                num_w = conv.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv.weight)
                conv.weight.data.copy_(conv_w)
                ptr += num_w
                conv = None

        else:
            # must solve the remained conv before encouter other modules
            if conv is not None:
                if conv.bias is not None:
                    num_b = conv.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv.bias)
                    conv.bias.data.copy_(conv_b)
                    ptr += num_b

                num_w = conv.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv.weight)
                conv.weight.data.copy_(conv_w)
                ptr += num_w
                conv = None

            if isinstance(module, nn.Linear):
                linear_layer = module
                num_b = linear_layer.bias.numel()
                linear_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(linear_layer.bias)
                linear_layer.bias.data.copy_(linear_b)
                ptr += num_b

                num_w = linear_layer.weight.numel()
                linear_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(linear_layer.weight)
                linear_layer.weight.data.copy_(linear_w)
                ptr += num_w

