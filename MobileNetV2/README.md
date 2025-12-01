# MobileNetV2

## Model infomation
This folder provides the code needed to reproduce the MobileNetV2 image classification model from pytorchcv.

| Name | Task | Source | FP32 | W8A8 |
| ---- | ---- | ---- | ---- | ---- |
| MobileNetV2 | Image Classification | https://github.com/osmr/pytorchcv | pytorchcv v0.0.67 | Model/model.tflite |

| Precision Mode | Format | Metric (Acc) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | Torch module | 73.044 | N/A |
| W8A8 | TFLite v2.17.0 | 72.994 | v1.1.1 |


## License
[MIT license](https://github.com/osmr/pytorchcv?tab=MIT-1-ov-file)


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
dummy_input: [[1,3,224,224]]
batch_size: 32
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
