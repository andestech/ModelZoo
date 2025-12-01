# DeepFilterNet

## Model infomation
This folder provides the code needed to reproduce the DeepFilterNet noise suppresion model described in the [DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering](https://arxiv.org/abs/2110.05588) paper.

We modified the model to cut off output postprocessing and fix input time dimension to 200, then convert ONNX format based on tensorflow-onnx tools. 

For evaluation, we use dataset from https://datashare.ed.ac.uk/handle/10283/2791 same as RNNoise example.

| Name | Task | Source | FP32 | W8A8 |
| ---- | ---- | ---- | ---- | ---- |
| DeepFilterNet | Noise Suppresion | https://github.com/Rikorose/DeepFilterNet | Model/sim_deepfilter_200.onnx | Model/model.tflite |

| Precision Mode | Format | Metric (100 wav files PESQ) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | ONNX (opset_v13) | 2.82 | N/A |
| W8A8 | TFLite v2.17.0 | 2.793 (AndesQuant result) | v1.1.1 |


## License
- [Apache-2.0](https://github.com/Rikorose/DeepFilterNet?tab=Apache-2.0-2-ov-file)
- [MIT license](https://github.com/Rikorose/DeepFilterNet?tab=MIT-3-ov-file)


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
|Symm| x  | v  | v  | x  |
|Asym| x  | v  | v  | x  |
