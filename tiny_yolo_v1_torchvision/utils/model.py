from typing import Any, Type, List
import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["create_network", "forward"]



class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size=2):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = F.max_pool2d(padded_x, self.kernel_size, stride=1)
        return pooled_x


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert x.data.dim() == 4
        B, C, H, W = x.data.shape
        hs = self.stride
        ws = self.stride
        assert H % hs == 0, (
            "The stride " + str(self.stride) + " is not a proper divisor of height " + str(H)
        )
        assert W % ws == 0, (
            "The stride " + str(self.stride) + " is not a proper divisor of height " + str(W)
        )
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(-2, -3).contiguous()
        x = x.view(B, C, H // hs * W // ws, hs, ws)
        x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(-1, -2).contiguous()
        x = x.view(B, C, ws * hs, H // ws, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, C * ws * hs, H // ws, W // ws)
        return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

def create_network(_class: Type[nn.Module], blocks: List[dict]):
    models = nn.ModuleList()
    out_filters = []
    index = 0
    for block in blocks:
        if block.get("type") == "net":
            prev_filters = int(block.get("channels"))
            setattr(_class, "channels", int(block.get("channels")))
            setattr(_class, "input_width", int(block.get("width")))
            setattr(_class, "input_height", int(block.get("height")))
            setattr(_class, "learning_rate", float(block.get("learning_rate")))
            setattr(_class, "momentum", float(block.get("momentum")))
            setattr(_class, "decay", float(block.get("decay")))
            setattr(_class, "angle", int(block.get("angle", 0)))
            setattr(_class, "saturation", float(block.get("saturation")))
            setattr(_class, "exposure", float(block.get("exposure")))
            setattr(_class, "hue", float(block.get("hue")))
            continue

        elif block.get("type") == "convolutional":
            batch_normalize = int(block.get("batch_normalize"))
            filters = int(block.get("filters"))
            kernel_size = int(block.get("size"))
            stride = int(block.get("stride"))
            pad = int(block.get("pad"))
            activation = block.get("activation")

            if pad:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            model = nn.Sequential()
            if batch_normalize:
                conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False)
                model.add_module("conv{0}".format(index), conv)
                model.add_module("bn{0}".format(index), nn.BatchNorm2d(filters))
            else:
                conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad)
                model.add_module("conv{0}".format(index), conv)

            if activation == "leaky":
                model.add_module("leaky{0}".format(index), nn.LeakyReLU(0.1, inplace=True))
            elif activation == "relu":
                model.add_module("relu{0}".format(index), nn.ReLU(inplace=True))

            prev_filters = filters
            out_filters.append(prev_filters)
            models.append(model)

        elif block["type"] == "upsample":
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="nearest")
            out_filters.append(prev_filters)
            models.append(upsample)

        elif block.get("type") == "maxpool":
            pool_size = int(block.get("size"))
            stride = int(block.get("stride"))
            if stride != 1:
                model = nn.MaxPool2d(pool_size, stride)
            else:
                model = MaxPoolStride1()
            out_filters.append(prev_filters)
            models.append(model)

        elif block.get("type") == "avgpool":
            model = GlobalAvgPool2d()
            out_filters.append(prev_filters)
            models.append(model)

        elif block.get("type") == "reorg":
            stride = int(block.get("stride"))
            prev_filters = stride * stride * prev_filters
            out_filters.append(prev_filters)
            models.append(Reorg(stride))

        elif block.get("type") == "route":
            _layers = block.get("layers").split(",")
            ind = len(models)
            _layers = [int(i) if int(i) > 0 else int(i) + ind for i in _layers]
            if len(_layers) == 1:
                prev_filters = out_filters[_layers[0]]
            elif len(_layers) == 2:
                assert _layers[0] == ind - 1
                prev_filters = out_filters[_layers[0]] + out_filters[_layers[1]]
            out_filters.append(prev_filters)
            models.append(EmptyModule())

        elif block.get("type") == "shortcut":
            ind = len(models)
            prev_filters = out_filters[ind - 1]
            out_filters.append(prev_filters)
            models.append(EmptyModule())

        elif block.get("type") == "connected":
            filters = int(block.get("output"))
            if block.get("activation") == "linear":
                # only for yolov1
                model = nn.Sequential(nn.Flatten(), nn.Linear(prev_filters * 7 * 7, filters))
            elif block.get("activation") == "leaky":
                model = nn.Sequential(nn.Linear(prev_filters, filters), nn.LeakyReLU(0.1, inplace=True))
            elif block.get("activation") == "relu":
                model = nn.Sequential(nn.Linear(prev_filters, filters), nn.ReLU(inplace=True))
            prev_filters = filters
            out_filters.append(prev_filters)
            models.append(model)

        elif block.get("type") == "detection":
            setattr(_class, "num_classes", int(block.get("classes")))
            setattr(_class, "num_bboxes", int(block.get("num")))

        elif block.get("type") == "region":
            anchors = block.get("anchors").split(",")
            anchors = [(float(anchors[i]), float(anchors[i + 1])) for i in range(0, len(anchors), 2)]
            setattr(_class, "anchors", anchors)
            setattr(_class, "num_classes", int(block.get("classes")))
            setattr(_class, "num_bboxes", int(block.get("num")))
            setattr(_class, "ignore_threshold", float(block.get("thresh")))

            out_filters.append(prev_filters)
            models.append(EmptyModule())

        elif block.get("type") == "yolo":
            mask = block.get("mask").split(",")
            mask = [int(x) for x in mask]
            anchors = block.get("anchors").split(",")
            anchors = [(int(anchors[i]), int(anchors[i + 1])) for i in range(0, len(anchors), 2)]
            setattr(_class, "anchors", anchors)
            setattr(_class, "num_classes", int(block.get("classes")))
            setattr(_class, "num_bboxes", int(block.get("num")))
            setattr(_class, "ignore_threshold", float(block.get("ignore_thresh")))
            out_filters.append(prev_filters)
            models.append(EmptyModule())

        else:
            raise Exception("unknown type %s" % (block.get("type")))

        index += 1
    return models


def forward(blocks: List[dict], models: nn.Module, x: Tensor):
    ind = -2
    outputs = dict()
    detections = []

    for block in blocks:
        ind = ind + 1

        if block.get("type") == "net":
            continue

        elif (
            block.get("type") == "convolutional"
            or block.get("type") == "upsample"
            or block.get("type") == "maxpool"
            or block.get("type") == "reorg"
            or block.get("type") == "avgpool"
            or block.get("type") == "softmax"
            or block.get("type") == "connected"
        ):
            x = models[ind](x)
            outputs[ind] = x

        elif block.get("type") == "route":
            layers = block.get("layers").split(",")
            layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
            if len(layers) == 1:
                x = outputs[layers[0]]
                outputs[ind] = x
            elif len(layers) == 2:
                x1 = outputs[layers[0]]
                x2 = outputs[layers[1]]
                x = torch.cat((x1, x2), 1)
                outputs[ind] = x

        elif block.get("type") == "shortcut":
            from_layer = int(block.get("from"))
            activation = block.get("activation")
            from_layer = from_layer if from_layer > 0 else from_layer + ind
            x1 = outputs[from_layer]
            x2 = outputs[ind - 1]
            x = x1 + x2
            if activation == "leaky":
                x = F.leaky_relu(x, 0.1, inplace=True)
            elif activation == "relu":
                x = F.relu(x, inplace=True)
            outputs[ind] = x

        elif block.get("type") == "detection":
            detections = x
            outputs[ind] = x

        elif block.get("type") == "region":
            detections = x.permute(0, 2, 3, 1)
            outputs[ind] = x

        elif block.get("type") == "yolo":
            detections.append(x.permute(0, 2, 3, 1))
            outputs[ind] = x
    return detections
