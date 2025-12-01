# ResNet-8

## Model infomation
This folder provides the code needed to reproduce the ResNet-8 image classification model described in the [MLPerf Tiny Benchmark](https://arxiv.org/pdf/2106.07597) paper.

| Name | Task | Source | FP32 | W8A8 |
| ---- | ---- | ---- | ---- | ---- |
| ResNet-8 | Image Classification | https://github.com/mlcommons/tiny | Model/ResNet8_fp32.onnx | Model/model.tflite |

| Precision Mode | Format | Metric (Acc) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | ONNX (opset_v13) | 87.18 | N/A |
| W8A8 | TFLite v2.17.0 | 86.96 | v1.1.1 |


## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)


## Dataset build
Follwoing the intruction from follwing URL to build ML-Perf specified cifar10 dataset:
- https://github.com/mlcommons/tiny/tree/master/benchmark/training/image_classification

The above URL would download cifar10 dataset from
- https://www.cs.toronto.edu/~kriz/cifar.html

And extract the calibration dataset with perf_samples_loader.py

(Remember to change target npy from perf_samples_idxs.npy to calibration_samples_idxs.npy) 


## Change model_cfg.yaml
Revise path in following model_cfg.yaml to your path
```sh
tra_dataset_path: "/your_path/cifar10"
val_dataset_path: "/your_path/cifar10"
cal_dataset_path: "/your_path/cifar10_cal"
batch_size: 128
dummy_input: [[1,3,32,32]]
channel: 3
height: 32
width: 32
fp32_min: 0.0
fp32_max: 255.0
```
The dataset setting is done.


## Available workflow
|    |Prun|SVD |PTQ |QAT |
|----|----|----|----|----|
|Symm| v  | v  | v  | v  |
|Asym| v  | v  | v  | v  |



