# RetinaFace

##  Model infomation
This folder provides the code needed to reproduce the RetinaFace face detection model described in the [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641) paper.

We fix input resolution into 480x480 and convert model from tflite to ONNX by tensorflow-onnx tool.

| Name | Task | Source | FP32 | W8A8 |
| ---- | ---- | ---- | ---- | ---- |
| RetinaFace | Face Detection | https://github.com/peteryuX/retinaface-tf2 | Model/simp_480.onnx | Model/model.tflite |

| Precision Mode | Format | Metric (mAP) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | ONNX (opset_v13) | 84.67 | N/A |
| W8A8 | TFLite v2.17.0 | 84.25 | v1.1.1 |


## License
[MIT license](https://github.com/peteryuX/retinaface-tf2?tab=License-1-ov-file)


## Dataset build
Refer to https://github.com/peteryuX/retinaface-tf2?tab=readme-ov-file#data-preparing. 

Following Data Preparing part to download and extract widerface dataset until Step 3 where directory strcutue has been created.

The final dataset structure should shows as follows:
```
/your_path/widerface/
    ├── train/
    │   ├── images/
    │   └── label.txt
    └── val/
        ├── images/
        └── label.txt
```


## Change model_cfg.yaml
Revise path in following model_cfg.yaml to your path
```sh
tra_dataset_path: "/yout_path/widerface/train"
val_dataset_path: "/yout_path/widerface/val"
batch_size: 8
dummy_input: [[1,3,480,480]]
channel: 3
height: 480
width: 480
fp32_min: -1.0
fp32_max: 1.0
image_size: [480,480]
min_sizes: [[16, 32], [64, 128], [256, 512]]
steps: [8,16,32]
match_thresh: 0.45
ignore_thresh: 0.3
variances: [0.1, 0.2]
clip: False
```
The dataset setting is done.


## Available workflow
|    |Prun|SVD |PTQ |QAT |
|----|----|----|----|----|
|Symm| v  | v  | v  | x  |
|Asym| v  | v  | v  | x  |
