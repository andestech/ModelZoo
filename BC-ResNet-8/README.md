# BC-ResNet-8

## Model infomation
This folder provides the code needed to reproduce the BC-ResNet-8 keyword spotting model described in the [Broadcasted Residual Learning for Efficient Keyword Spotting](https://arxiv.org/abs/2106.04140) paper.

| Name | Task | Source | FP32 | W8A8 |
| ---- | ---- | ---- | ---- | ---- |
| BC-ResNet-8 | Keyword Spotting | https://github.com/Qualcomm-AI-research/bcresnet | Model/bcresnet_fp32.pth | Model/model.tflite |

| Precision Mode | Format | Metric (Acc) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | Torch module | 98.81391 | N/A |
| W8A8 | TFLite v2.17.0 | 98.56851 | v1.1.1 |


## License
[BSD-3-Clause-Clear license](https://github.com/Qualcomm-AI-research/bcresnet?tab=BSD-3-Clause-Clear-1-ov-file)


## Dataset build
The Speech Commands dataset can be downloaded from the following link:
- https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz

Extract the archive file after downloading and use the SplitDataset function located in utils.py to generate the training, validation, and testing folders for the 12-class dataset.

Once the process is complete, collect the following folders — \_background_noise\_, train_12class, valid_12class, test_12class — and place them together under a single parent directory.

The final structure show as following:
```
/your_path/speech_commands_v0.02_split/
    ├── _background_noise_/
    ├── train_12class/
    │   ├── _silence_/
    │   ├── _unknown_/
    │   ├── down/
    │   ├── go/
    │   ├── left/
    │   ├── no/
    │   ├── off/
    │   ├── on/
    │   ├── right/
    │   ├── stop/
    │   ├── up/
    │   └── yes/
    ├── valid_12class/
    └── test_12class/
```
Each of the folders train_12class, valid_12class, and test_12class contains the same subdirectory structure for the 12 classes listed above.


## Change model_cfg.yaml
Revise path in following model_cfg.yaml to your path
```sh
tra_dataset_path: "/your_path/speech_commands_v0.02_split/train_12class"
val_dataset_path: "/your_path/speech_commands_v0.02_split/valid_12class"
test_dataset_path: "/your_path/speech_commands_v0.02_split/test_12class"
noise_dataset_path: "/your_path/speech_commands_v0.02_split/_background_noise_"
batch_size: 128
dummy_input: [[1,1,40,101]]
channel: 3
height: 32
width: 32
fp32_min: -14.0
fp32_max: 9.0
```
The dataset setting is done.


## Available workflow
|    |Prun|SVD |PTQ |QAT |
|----|----|----|----|----|
|Symm| x  | v  | v  | v  |
|Asym| x  | v  | v  | v  |



