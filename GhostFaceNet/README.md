# GhostFaceNet

## Model infomation
This folder provides the code needed to reproduce the GhostFaceNet face recognition model described in the [GhostFaceNets: Lightweight Face Recognition Model From Cheap Operations](https://ieeexplore.ieee.org/document/10098610) paper.

| Name | Task | Source | FP32 | W8A8 |
| ---- | ---- | ---- | ---- | ---- |
| GhostFaceNet | Face Recognition | https://github.com/HamadYA/GhostFaceNets | Model/ghostface_fp32.onnx | Model/model.tflite |

| Precision Mode | Format | Metric (Acc) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | ONNX (opset_v15) | 99.683 | N/A |
| W8A8 | TFLite v2.17.0 | 99.450 | v1.1.1 |


## License
[MIT license](https://github.com/HamadYA/GhostFaceNets?tab=MIT-1-ov-file)


## Dataset build
Follow the instructions from the official GhostFaceNet repository to build the specified face recognition dataset:
- https://github.com/HamadYA/GhostFaceNets?tab=readme-ov-file#datasets-and-data-preparation

For validation dataset, use the LFW binary dataset (lfw.bin) and convert it into a PyTorch .pt format dataset.

The final dataset structure should shows as follows:
```
/your_path/GhostFace_dataset/
    ├── faces_emore_112x112_folders/
    │   ├── 0
    │   │   ├── 1.jpg
    │   │   ├── 2.jpg
    │   │   ├── 3.jpg
    │   ├── 1
    │   │   ├── 111.jpg
    │   │   ├── 112.jpg
    │   │   ├── 113.jpg
    │   ├── 10
    │       ├── 707.jpg
    │       ├── 708.jpg
    │       ├── 709.jpg
    │   
    └── valid_data/
        ├── input_0.pt
        ├── input_1.pt
        ├── input_2.pt
```



## Change model_cfg.yaml
Revise path in following model_cfg.yaml to your path
```sh
tra_dataset_path: "/your_path/GhostFace_dataset/faces_emore_112x112_folders"
val_dataset_path: "/your_path/GhostFace_dataset/valid_data"
batch_size: 128
dummy_input: [[1,3,112,112]]
channel: 3
height: 32
width: 32
fp32_min: -0.9961
fp32_max: 0.9961
```
The dataset setting is done.


## Available workflow
|    |Prun|SVD |PTQ |QAT |
|----|----|----|----|----|
|Symm| v  | v  | v  | v  |
|Asym| v  | v  | v  | v  |



