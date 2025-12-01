# MCUNet-VWW2

## Model infomation
This folder provides the code needed to reproduce the MCUNet-VWW2 model described in the [MCUNet: Tiny Deep Learning on IoT Devices](https://arxiv.org/abs/2007.10319) paper.

| Name | Task | Source | FP32 | W8A8 |
| ---- | ---- | ---- | ---- | ---- |
| MCUNet-VWW2 | Visual Wake Words | https://github.com/mit-han-lab/mcunet | Model/mcunet | Model/model.tflite |

| Precision Mode | Format | Metric (Acc) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | Torch module | 91.87244 | N/A |
| W8A8 | TFLite v2.17.0 | 91.63668 | v1.1.1 |


## License
[MIT license](https://github.com/mit-han-lab/mcunet?tab=MIT-1-ov-file)


## Dataset build
Follow the instructions used in pyvww to download and extract the Visual Wake Words dataset.
- https://pypi.org/project/pyvww/

The final structure show as following:
```
/your_path/visualwakewords/coco/
    ├── all/                     # Contains all COCO 2014 training and validation images
    |   ├── COCO_train2014_000000000009.jpg
    |   ├── COCO_val2014_000000000042.jpg
    |   ├── ...
    └── annotations/             # Contains COCO annotation files
        ├── instances_train.json
        └── instances_val.json
```


## Change model_cfg.yaml
Revise path in following model_cfg.yaml to your path
```sh
data_source_tra: "/your_path/visualwakewords/coco/all"
data_source_val: "/your_path/visualwakewords/coco/all"
ann_file_tra: "/your_path/visualwakewords/coco/annotations/instances_train.json"
ann_file_val: "/your_path/visualwakewords/coco/annotations/instances_val.json"
batch_size: 64
dummy_input: [[1,3,144,144]]
channel: 3
width: 144
height: 144
fp32_min: -1.0
fp32_max: 1.0
```
The dataset setting is done.


## Available workflow
|    |Prun|SVD |PTQ |QAT |
|----|----|----|----|----|
|Symm| v  | v  | v  | v  |
|Asym| v  | v  | v  | v  |



