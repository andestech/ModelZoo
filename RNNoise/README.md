# RNNoise

## Model infomation
This folder provides the code needed to reproduce the RNNoise noise suppresion model described in the [A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement](https://arxiv.org/abs/1709.08243) paper.

We modified the model to cut off input preprocessing and output postprocessing and convert ONNX format based on tensorflow-onnx tools

| Name | Task | Source | FP32 | W8A8 |
| ---- | ---- | ---- | ---- | ---- |
| RNNoise | Noise Suppresion | https://github.com/xiph/rnnoise | Model/rnnoise_fp32.onnx | Model/model.tflite |

| Precision Mode | Format | Metric (PESQ) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | ONNX (opset_v13) | 2.461 | N/A |
| W8A8 | TFLite v2.17.0 | 2.434 | v1.1.1 |


## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)


## Dataset build
The dataset is accessible for download at the following link: 
- https://datashare.ed.ac.uk/handle/10283/2791

In the website download following as validation data:
- clean_testset_wav.zip (147.1Mb)
- noisy_testset_wav.zip (162.6Mb)

Download following as training data:
- clean_trainset_56spk_wav.zip (4.442Gb)
- noisy_trainset_56spk_wav.zip (5.240Gb)

Extract the zip file and the final structure show as following:
```
/your_path/rnnwave/
    ├── clean_trainset_56spk_wav/
    │   ├── p234_001.wav
    │   ├── p234_002.wav
    │   ├── p234_003.wav
    │
    ├── noisy_trainset_56spk_wav/
    ├── clean_testset_wav/
    └── noisy_testset_wav/
```
Each of the clean and noisy folders for the train and test data will contain multiple .wav audio files.


## Change model_cfg.yaml
Revise path in followin model_cfg.yaml to your path
```sh
train_clean_wave_path: "/your_path/rnnwave/clean_trainset_56spk_wav/"
train_noisy_wave_path: "/your_path/rnnwave/noisy_trainset_56spk_wav/"
clean_wave_path: "/your_path/rnnwave/clean_testset_wav"
noisy_wave_path: "/your_path/rnnwave/noisy_testset_wav"
dummy_input: [[1,1,42],[1,24],[1,48],[1,96]]
batch_size: 1
channel: 1
height: 42
width: 1
fp32_min: -41.1027
fp32_max: 41.2633
```
The dataset setting is done.


## Available workflow
|    |Prun|SVD |PTQ |QAT |
|----|----|----|----|----|
|Symm| x  | v  | v  | v  |
|Asym| x  | v  | v  | v  |
