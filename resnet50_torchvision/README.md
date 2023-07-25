# Instruction to navigate dataset to dataset.py

Download ILSVRC2012 IMAGENET dataset from https://www.image-net.org/download.php

Account is required and verfied by 2020 Stanford Vision Lab

Extract the zip and the structure should show as following
```
ILSVRC2012/
  --raw-data/
    --imagenet-data/
        --bounding_boxes/
        --logs/
        --raw-data/
        --train/
        --val/
```
Modify dataset path to user own path in model_cfg.yaml.

Example:

tra_dataset_path: "/dataset/ILSVRC2012/raw-data/imagenet-data/train"

val_dataset_path: "/dataset/ILSVRC2012/raw-data/imagenet-data/val"
