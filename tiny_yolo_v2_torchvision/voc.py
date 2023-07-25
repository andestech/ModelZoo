from typing import Callable, Dict, List, Tuple
import sys

import numpy as np
from PIL.Image import Image
from model_ws_package.tiny_yolo_v2_torchvision.transforms import *
from model_ws_package.tiny_yolo_v2_torchvision.utils import YOLOv2TrainPostProcessing
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import VOCDetection

__all__ = [
    "VOC_CLASS",
    "NUM_CLASSES",
    "voc_detection",
    "voc_detection_yolov2",
]

VOC_CLASS = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
NUM_CLASSES = len(VOC_CLASS)


class LabelPreProcessing:
    def __call__(self, img: Image, label: dict) -> Tuple[Image, List[np.ndarray]]:
        objs_info = label["annotation"]["object"]

        target_bboxes = []
        target_classes = []
        target_difficults = []
        for info in objs_info:
            category = info["name"]
            difficult = int(info["difficult"])
            xmin = float(info["bndbox"]["xmin"]) - 1
            xmax = float(info["bndbox"]["xmax"]) - 1
            ymin = float(info["bndbox"]["ymin"]) - 1
            ymax = float(info["bndbox"]["ymax"]) - 1
            target_bboxes.append([xmin, ymin, xmax, ymax])
            target_classes.append(VOC_CLASS.index(category))
            target_difficults.append(difficult)

        img = np.array(img)
        target_bboxes = np.array(target_bboxes)
        target_classes = np.array(target_classes)
        target_difficults = np.array(target_difficults)
        return img, [target_bboxes, target_classes, target_difficults]


class TestPostProcessing:
    def __call__(self, img: np.ndarray, target: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        height, width, _ = img.shape
        norm_bboxes = target[0] / np.array([width, height, width, height])
        norm_bboxes[..., :2] = (norm_bboxes[..., :2] + norm_bboxes[..., 2:]) / 2
        norm_bboxes[..., 2:] = (norm_bboxes[..., 2:] - norm_bboxes[..., :2]) * 2
        target[0] = norm_bboxes
        return img, target


def _get_voc_dataset(
    root: str,
    year: str,
    download: bool = False,
    transform: dict = {},
    target_transform: Callable = None,
    transforms: dict = {},
):
    if year == "2012":
        train_dataset = VOCDetection(
            root=root,
            year="2012",
            image_set="train",
            download=download,
            transform=transform.get("train"),
            target_transform=target_transform,
            transforms=transforms.get("train"),
        )
        val_dataset = VOCDetection(
            root=root,
            year="2012",
            image_set="val",
            download=download,
            transform=transform.get("val"),
            target_transform=target_transform,
            transforms=transforms.get("val"),
        )
        test_dataset = VOCDetection(
            root=root,
            year="2012",
            image_set="val",
            download=download,
            transform=transform.get("test"),
            target_transform=target_transform,
            transforms=transforms.get("test"),
        )
    elif year == "2007":
        train_dataset = VOCDetection(
            root=root,
            year="2007",
            image_set="train",
            download=download,
            transform=transform.get("train"),
            target_transform=target_transform,
            transforms=transforms.get("train"),
        )
        val_dataset = VOCDetection(
            root=root,
            year="2007",
            image_set="val",
            download=download,
            transform=transform.get("val"),
            target_transform=target_transform,
            transforms=transforms.get("val"),
        )
        test_dataset = VOCDetection(
            root=root,
            year="2007",
            image_set="test",
            download=download,
            transform=transform.get("test"),
            target_transform=target_transform,
            transforms=transforms.get("test"),
        )
    elif year == "2007+2012" or year == "2012+2007":
        train_dataset = ConcatDataset(
            [
                VOCDetection(
                    root=root,
                    year="2007",
                    image_set="trainval",
                    download=download,
                    transform=transform.get("train"),
                    target_transform=target_transform,
                    transforms=transforms.get("train"),
                ),
                VOCDetection(
                    root=root,
                    year="2012",
                    image_set="trainval",
                    download=download,
                    transform=transform.get("train"),
                    target_transform=target_transform,
                    transforms=transforms.get("train"),
                ),
            ]
        )
        val_dataset = ConcatDataset(
            [
                VOCDetection(
                    root=root,
                    year="2007",
                    image_set="val",
                    download=download,
                    transform=transform.get("val"),
                    target_transform=target_transform,
                    transforms=transforms.get("val"),
                ),
                VOCDetection(
                    root=root,
                    year="2012",
                    image_set="val",
                    download=download,
                    transform=transform.get("val"),
                    target_transform=target_transform,
                    transforms=transforms.get("val"),
                ),
            ]
        )
        test_dataset = VOCDetection(
            root=root,
            year="2007",
            image_set="test",
            download=download,
            transform=transform.get("test"),
            target_transform=target_transform,
            transforms=transforms.get("test"),
        )
    print(len(test_dataset))
    print(len(val_dataset))

    return train_dataset, val_dataset, test_dataset


def voc_detection(
    postprocessing: Callable[[np.ndarray, List[np.ndarray]], Tuple[np.ndarray, List[np.ndarray]]],
    root: str = "/dataset",
    year: str = "2007+2012",
    img_size: tuple = (448, 448),
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 6,
    pin_memory: bool = False,
    prefetch_factor: int = 5,
) -> Dict[str, DataLoader]:
    dataloaders = {}
    data_transforms = {
        "train": TransformsCompose(
            [
                LabelPreProcessing(),
                RandomBrightness((0.75, 1)),
                RandomSaturation((0.75, 1)),
                RandomHue((0.75, 1)),
                # RandomBlur(),
                RandomShift(),
                # RandomScale(),
                RandomFlip(),
                RandomResizedCrop(img_size),
                postprocessing,
                ToTensor(),
            ]
        ),
        "val": TransformsCompose(
            [
                LabelPreProcessing(),
                Resize(img_size),
                postprocessing,
                ToTensor(),
            ]
        ),
        "test": TransformsCompose(
            [
                LabelPreProcessing(),
                Resize(img_size),
                TestPostProcessing(),
                ToTensor(),
            ]
        ),
    }
    train_dataset, val_dataset, test_dataset = _get_voc_dataset(
        root,
        year,
        download=True,
        transforms=data_transforms,
    )
    dataloaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    dataloaders["val"] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    dataloaders["test"] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    dataloaders["cos"] = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    return dataloaders


def voc_detection_yolov2(
    root: str = "/dataset",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 6,
    pin_memory: bool = False,
    prefetch_factor: int = 5,
):
    return voc_detection(
        postprocessing=YOLOv2TrainPostProcessing(),
        root=root,
        year="2007+2012",
        img_size=(416, 416),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
