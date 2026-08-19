# TinyBERT-SQuADv2

## Model infomation
This folder provides the code needed to reproduce the TinyBERT model with SQuADv2 task.

| Name | Task | Source | FP32 | W8A8 qmode-2 |
| ---- | ---- | ---- | ---- | ---- |
| TinyBERT-SQuADv2 | Extractive Question Answering | https://huggingface.co/deepset/tinybert-6l-768d-squad2 | https://huggingface.co/deepset/tinybert-6l-768d-squad2 | Model/model.tflite |

| Precision Mode | Format | Metric (Acc) | NN SDK Version |
| ---- | ---- | ---- | ---- |
| FP32 | Torch module | 77.18 | N/A |
| W8A8 qmode-2 | TFLite v2.17.0 | 76.18 | v1.2.0 |


## License
[MIT license](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)


## Dataset build
The dataset Download API is written in dataset.py 
```
from datasets import load_dataset

raw_datasets = load_dataset(
    "squad_v2",
    None,
    cache_dir=input_yaml['dataset_path'],
    token=None,
    trust_remote_code=False
)
```
Prepare an extist directiory "/your_path/squadv2_dataset"

Set path for key "dataset_path" in model_cfg.yaml



## Change model_cfg.yaml
Revise path in following model_cfg.yaml to your path
```sh
dataset_path: "/your_path/squadv2_dataset"
batch_size: 13
dummy_input: [[1,384],[1,384],[1,384]]
input_type: ['torch.int32','torch.float32','torch.int32']
channel: 3
width: 224
height: 224
fp32_min: -2.1179
fp32_max: 2.64
```
The dataset setting is done.


## Available workflow
|    |Prun|SVD |PTQ |QAT |
|----|----|----|----|----|
|Symm| v  | v  | v  | v  |
|Asym| v  | v  | v  | v  |
