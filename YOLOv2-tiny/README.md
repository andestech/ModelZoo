# YOLOv2-tiny

##  Model infomation
This folder provides the code needed to reproduce the YOLOv2-tiny object detection model described in the [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) paper.

We serve nonmaxsuppression as postprocessing. Model weight re-train by ourselves.

| Name | Task | Source | FP32 | W8A8 |
| ---- | ---- | ---- | ---- | ---- |
| YOLOv2-tiny | Object Detection | https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny.cfg | Model/yolov2-tiny-voc.weights | Model/model.tflite |

| Precision Mode | Format | Metric (mAP) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | Torch module | 56.84 | N/A |
| W8A8 | TFLite v2.17.0 | 56.24 | v1.1.1 |


## License
[MIT license](https://github.com/pjreddie/darknet?tab=MIT-6-ov-file)


## Dataset build
This model workspace package would download the dataset VOCdevkit itself.

Specify your dataset path in model_cfg.yaml.
```
VOCdevkit_root_path: "/your_path"
```

The dataset would download tar and locate at
```
"/your_path/VOCdevkit/"
```

dataset script would check the following would exist:
```
VOC2012test.tar
VOCtest_06-Nov-2007.tar
VOCtrainval_06-Nov-2007.tar
VOCtrainval_11-May-2012.tar
```
and extract them to right path automatically.


## Change model_cfg.yaml
Revise path in following model_cfg.yaml to your path
```sh
VOCdevkit_root_path: "/your_path/VOCdevkit"
batch_size: 64
dummy_input: [[1,3,416,416]]
channel: 3
height: 416
width: 416
fp32_min: 0.0
fp32_max: 1.0
```
The dataset setting is done.


## Available workflow
|    |Prun|SVD |PTQ |QAT |
|----|----|----|----|----|
|Symm| v  | v  | v  | x  |
|Asym| v  | v  | v  | v  |

