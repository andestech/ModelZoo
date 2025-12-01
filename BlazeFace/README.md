# BlazeFace

## Model infomation
This folder provides the code needed to reproduce the BlazeFace face detection model described in the [BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs](https://arxiv.org/abs/1907.05047) paper.

| Name | Task | Source | FP32 | W8A8 |
| ---- | ---- | ---- | ---- | ---- |
| BlazeFace | Face Detection | https://github.com/zineos/blazeface | Model/blazeface_fp32.pth | Model/model.tflite |

| Precision Mode | Format | Metric (mAP) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | Torch module | 87.71012 | N/A |
| W8A8 | TFLite v2.17.0 | 87.52426 | v1.1.1 |


## License


## Dataset build
Follwoing the intruction from follwing URL to download and extract the widerface dataset:
- https://github.com/zineos/blazeface?tab=readme-ov-file#installation

The final dataset structure should shows as follows:
```
/your_path/widerface/
    ├── train/
    │   ├── images/
    │   │   ├── 0--Parade/
    │   │   ├── 1--Handshaking/
    │   │   ├── 10--People_Marching/
    │   │
    │   └── label.txt
    └── val/
        ├── images/
        │   ├── 0--Parade/
        │   ├── 1--Handshaking/
        │   ├── 10--People_Marching/
        │
        ├── timer.py
        └── wider_val.txt
```


## Change model_cfg.yaml
Revise path in following model_cfg.yaml to your path
```sh
trainset_folder : "/your_path/widerface/train/images/"
testset_folder : "/your_path/widerface/val/images/"
dummy_input: [[1,3,640,640]]
batch_size: 4
channel: 3
width: 640
height: 640
fp32_min: -123.0
fp32_max: 151.0
```
The dataset setting is done.


## Available workflow
|    |Prun|SVD |PTQ |QAT |
|----|----|----|----|----|
|Symm| v  | v  | v  | v  |
|Asym| v  | v  | v  | v  |



