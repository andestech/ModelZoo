# MobileNetV1-SSD

## Model infomation
This folder provides the code needed to reproduce the MobileNetV1-SSD object detection model described in the [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) paper.

| Name | Task | Source | FP32 | W8A8 |
| ---- | ---- | ---- | ---- | ---- |
| MobileNetV1-SSD | Object Detection | https://github.com/qfgaohao/pytorch-ssd | Model/SSDs/weights/mobilenet-v1-ssd-mp-0_675.pth | Model/model.tflite |

| Precision Mode | Format | Metric (mAP) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | Torch module | 67.59754 | N/A |
| W8A8 | TFLite v2.17.0 | 67.69373 | v1.1.1 |


## License
[MIT license](https://github.com/qfgaohao/pytorch-ssd?tab=MIT-1-ov-file)


## Dataset build
Download and extract the PASCAL VOC dataset using the following script.
```sh
#!/bin/bash
# Download Images
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# Unzip
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOCtest_06-Nov-2007.tar
```

The VOCdevkit directory is generated and contains folllowing structure.
```
/your_path/VOCdevkit/
    ├── VOC2007
    |   ├── Annotations
    |   ├── ImageSets
    |   |   ├── Layout
    |   |   ├── Main
    |   |   └── Segmentation
    |   ├── JPEGImages
    |   ├── SegmentationClass
    |   └── SegmentationObject
    └── VOC2012
        ├── Annotations
        ├── ImageSets
        |   ├── Action
        |   ├── Layout
        |   ├── Main
        |   └── Segmentation
        ├── JPEGImages
        ├── SegmentationClass
        └── SegmentationObject
```


## Change model_cfg.yaml
Revise path in following model_cfg.yaml to your path
```sh
tra_dataset_path: "/your_path/VOCdevkit/VOC2012/"
val_dataset_path: "/your_path/VOCdevkit/VOC2007/"
batch_size: 64
dummy_input: [[1,3,300,300]]
channel: 3
width: 300
height: 300
fp32_min: -1.0
fp32_max: 1.0
```
The dataset setting is done.


## Available workflow
|    |Prun|SVD |PTQ |QAT |
|----|----|----|----|----|
|Symm| v  | v  | v  | v  |
|Asym| v  | v  | v  | v  |
