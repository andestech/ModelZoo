# DS-CNN-L

## Model infomation
This folder provides the code needed to reproduce the DS-CNN-L keyword spotting model described in the [Hello Edge: Keyword Spotting on Microcontrollers](https://arxiv.org/pdf/1711.07128) paper.

| Name | Task | Source | FP32 | W8A8 |
| ---- | ---- | ---- | ---- | ---- |
| DS-CNN-L | Keyword Spotting | https://github.com/Arm-Examples/ML-zoo/tree/master/models/keyword_spotting/ds_cnn_large/model_package_tf | Model/DSCNN_L_fp32.onnx | Model/model.tflite |

| Precision Mode | Format | Metric (Acc) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | ONNX (opset_v13) | 95.03068 | N/A |
| W8A8 | TFLite v2.17.0 | 95.13292 | v1.1.1 |


## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)


## Dataset build
Following the instructions provided in the following URL to build the MLPerf-specified Keyword Spotting dataset: 
- https://github.com/mlcommons/tiny/tree/master/benchmark/training/keyword_spotting

The above repository will automatically download the Google Speech Commands v2 dataset from:
- http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

Extract the zip file and organize the dataset into the desired structure. Then, convert and save the processed data into numpy format for subsequent use.

The final structure show as following:
```
/your_path/dscnn_processed_dataset/
    ├── train/
    │   ├── 0/          # class 0
    │   │   ├── 0.npy
    │   │   ├── 1.npy
    │   │   ├── 2.npy
    │   │
    │   ├── 1/          # class 1
    │   ├── 2/          # class 2
    │
    ├── valid/
    └── test/
```
Each of the folders train, valid, and test contains the same subdirectory structure for the 12 classes keyword spotting dataset.


## Change model_cfg.yaml
Revise path in following model_cfg.yaml to your path
```sh
tra_dataset_path: "/your_path/dscnn_processed_dataset/train"
val_dataset_path: "/your_path/dscnn_processed_dataset/val"
test_dataset_path: "/your_path/dscnn_processed_dataset/test"
batch_size: 128
dummy_input: [[1,1,49,10]]
channel: 1
width: 10
height: 49
fp32_min: -123.5697250366
fp32_max: 43.6676597595
```
The dataset setting is done.


## Available workflow
|    |Prun|SVD|PTQ|QAT|
|----|----|----|----|----|
|Symm| v  | v  | v  | v  |
|Asym| v  | v  | v  | v  |
