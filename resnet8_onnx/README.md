# Instruction to navigate dataset to dataset.py

Follwoing the intruction from follwing URL to build ML-Perf specified cifar10 dataset:

https://github.com/mlcommons/tiny/tree/master/benchmark/training/image_classification

The above URL would download cifar10 dataset from
 
https://www.cs.toronto.edu/~kriz/cifar.html

And extract the calibration dataset with perf_samples_loader.py
 
(Remember to change target npy from perf_samples_idxs.npy to calibration_samples_idxs.npy) 

Modified dataset path in model_cfg.yaml.

Example:
```
tra_dataset_path: "/dataset/cifar10"
val_dataset_path: "/dataset/cifar10"
cal_dataset_path: "/dataset/mlperf/ic/ic01_cal"     #Create by perf_samples_loader.py
```
