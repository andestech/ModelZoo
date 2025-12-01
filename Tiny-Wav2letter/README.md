# Tiny-Wav2letter

##  Model infomation
This folder provides the code needed to reproduce the Tiny-Wav2letter speech recogntion model described in the [Wav2Letter: an End-to-End ConvNet-based Speech Recognition System](https://arxiv.org/abs/1609.03193) paper.

Convert model from tflite FP32 to ONNX format based on tensorflow-onnx tool.

| Name | Task | Source | FP32 | W8A8 |
| ---- | ---- | ---- | ---- | ---- |
| Tiny-Wav2letter | Speech Recogntion | https://github.com/Arm-Examples/ML-zoo/tree/master/models/speech_recognition/tiny_wav2letter/tflite_int8/recreate_code | Model/tiny_wav2letter.onnx | Model/model.tflite |

| Precision Mode | Format | Metric (LER) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | ONNX (opset_v13) | 2.7977 | N/A |
| W8A8 | TFLite v2.17.0 | 3.0109 | v1.1.1 |


## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)


## Dataset build
Refer to https://github.com/Arm-Examples/ML-zoo/tree/master/models/speech_recognition/tiny_wav2letter/tflite_int8/recreate_code/preprocessing.py  building dataset folder.

The final structure show as following:
```
/your_path/wav2letter/
    ├── fluent_speech_commands_dataset/
    ├── librispeech_full_size/
    └── librispeech_reduced_size/
```


## Change model_cfg.yaml
Revise path in following model_cfg.yaml to your path
```sh
tra_dataset_path: "/your_path/wav2letter/"
batch_size: 128
dummy_input: [[1, 39, 1, 296]]
channel: 1
width: 296
height: 39
fp32_min: -5.5
fp32_max: 15.5
```
The dataset setting is done.


## Available workflow
|    |Prun|SVD |PTQ |QAT |
|----|----|----|----|----|
|Symm| v  | v  | v  | x  |
|Asym| v  | v  | v  | v  |

