# DTLN (Dual-signal Transformation LSTM Network)

## Model infomation
This folder provides the code needed to reproduce the DTLN noise suppression model described in the [Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression](https://www.isca-speech.org/archive/interspeech_2020/westhausen20_interspeech.html) paper.

|Name|Task|Source|FP32|W8A16|
|----|----|----|----|----|
| DTLN | Noise Suppresion | https://github.com/breizhn/DTLN | Model/DTLN_model1_fp32.onnx, Model/DTLN_model2_fp32.onnx | Model/model.tflite |

| Precision Mode | Format | Metric (PESQ) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | ONNX (opset_v13) | 3.15756 | N/A |
| W8A16 | TFLite v2.17.0 | 3.15354 (AndesQuant result) | v1.1.1 |


## License
[MIT license](https://github.com/breizhn/DTLN?tab=MIT-1-ov-file)


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
dummy_input: [[1,1,512],[1,2,128,2],[1,2,128,2]]
recursive_input: [[1,1],[2,2]]
batch_size: 1
channel: 1
height: 257
width: 1
fp32_min: -413.9567
fp32_max: 413.9567
```
The dataset setting is done.


## Available workflow
|    |Prun|SVD |PTQ |QAT |
|----|----|----|----|----|
|Symm| x  | v  | v  | v  |
|Asym| x  | v  | v  | v  |



