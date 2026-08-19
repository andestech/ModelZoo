# Swin-Transformer

## Model infomation
This folder provides the code needed to reproduce the Swin-Transformer image classification model.

| Name | Task | Source | FP32 | W8A8 qmode-2 |
| ---- | ---- | ---- | ---- | ---- |
| Swin-Transformer | Image Classification | https://huggingface.co/microsoft/swin-tiny-patch4-window7-224 | https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/tree/main | Model/model.tflite |

| Precision Mode | Format | Metric (Acc) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | Torch module | 81.18 | N/A |
| W8A8 qmode-2 | TFLite v2.17.0 | 79.438 | v1.2.0 |


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
