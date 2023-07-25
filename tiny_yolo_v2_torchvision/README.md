# Instruction to navigate dataset to dataset.py

This model workspace package would download the dataset VOCdevkit itself.

Specify your dataset path in model_cfg.yaml.

Example:

VOCdevkit_root_path: "/dataset"

The dataset would download tar and locate at

"/dataset/VOCdevkit/"

dataset script would check the following would exist:
```
VOC2012test.tar
VOCtest_06-Nov-2007.tar
VOCtrainval_06-Nov-2007.tar
VOCtrainval_11-May-2012.tar
```
and extract them to right path automatically.

