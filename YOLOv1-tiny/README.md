# YOLOv1-tiny

##  Model infomation
This folder provides the code needed to reproduce the YOLOv1-tiny object detection model described in the [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) paper.

We serve nonmaxsuppression as postprocessing. Model weight re-train by ourselves.

| Name | Task | Source | FP32 | W8A8 |
| ---- | ---- | ---- | ---- | ---- |
| YOLOv1-tiny | Object Detection | https://github.com/pjreddie/darknet/blob/master/cfg/yolov1-tiny.cfg | Model/yolov1.pt | Model/model.tflite |

| Precision Mode | Format | Metric (mAP) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | Torch module | 46.58 | N/A |
| W8A8 | TFLite v2.17.0 | 46.98 | v1.1.1 |


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
dummy_input: [[1,3,448,448]]
channel: 3
height: 448
width: 448
fp32_min: 0.0
fp32_max: 1.0
```
The dataset setting is done.


## Available workflow
|    |Prun|SVD |PTQ |QAT |
|----|----|----|----|----|
|Symm| v  | v  | v  | x  |
|Asym| v  | v  | v  | v  |

