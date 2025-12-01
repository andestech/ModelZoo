# EfficientNet-Lite0

## Model infomation
This folder provides the code needed to reproduce the EfficientNet-Lite0 image classification model described in the [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) paper.

| Name | Task | Source | FP32 | W8A8 |
| ---- | ---- | ---- | ---- | ---- |
| EfficientNet-Lite0 | Image Classification | https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite | Model/efficient_lite0_fp32.onnx | Model/model.tflite |

| Precision Mode | Format | Metric (Acc) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | ONNX (opset_v13) | 74.866 | N/A |
| W8A8 | TFLite v2.17.0 | 74.872 | v1.1.1 |


## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)


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
batch_size: 64
dummy_input: [[1,3,224,224]]
channel: 3
width: 224
height: 224
fp32_min: -1.0
fp32_max: 1.0
```
The dataset setting is done.

## Available workflow
|    |Prun|SVD |PTQ |QAT |
|----|----|----|----|----|
|Symm| v  | v  | v  | v  |
|Asym| v  | v  | v  | v  |
