# ResNet-50

## Model infomation
This folder provides the code needed to reproduce the ResNet-50 image classification model from torchvision.

| Name | Task | Source | FP32 | W8A8 |
| ---- | ---- | ---- | ---- | ---- |
| ResNet-50 | Image Classification | https://github.com/pytorch/vision | https://download.pytorch.org/models/resnet50-0676ba61.pth | Model/model.tflite |

| Precision Mode | Format | Metric (Acc) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | Torch module | 76.152 | N/A |
| W8A8 | TFLite v2.17.0 | 75.968 | v1.1.1 |


## License
[BSD-3-Clause license](https://github.com/pytorch/vision/tree/main?tab=BSD-3-Clause-1-ov-file)


## Dataset build
Download ILSVRC2012 IMAGENET dataset from https://www.image-net.org/download.php

Account is required and verfied by 2020 Stanford Vision Lab

Extract the zip and the structure should show as following:
```
/your_path/ILSVRC2012/
    └── raw-data/
        └── imagenet-data/
            ├── bounding_boxes/
            ├── logs/
            ├── raw-data/
            ├── train/
            └── val/
```


## Change model_cfg.yaml
Revise path in following model_cfg.yaml to your path
```sh
tra_dataset_path: "/your_path/ILSVRC2012/raw-data/imagenet-data/train"
val_dataset_path: "/your_path/ILSVRC2012/raw-data/imagenet-data/val"
batch_size: 128
dummy_input: [[1,3,224,224]]
channel: 3
width: 224
height: 224
fp32_min: -2.1179
fp32_max: 2.64
```
The dataset setting is done.


## Available workflow
|    |Prun|SVD |PTQ |QAT |
|----|----|----|----|----|
|Symm| v  | v  | v  | v  |
|Asym| v  | v  | v  | v  |
