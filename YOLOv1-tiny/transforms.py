# ref: https://github.com/motokimura/yolo_v1_pytorch/blob/c3e60d7abdb6a36c472e4dae55ed696dfc08dd43/voc.py
import random
from typing import Callable, List, Tuple, Union

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from PIL.Image import Image

__all__ = [
    "TransformsCompose",
    "RandomBrightness",
    "RandomSaturation",
    "RandomHue",
    "RandomBlur",
    "SubMean",
    "ConvertColor",
    "RandomShift",
    "RandomScale",
    "RandomCrop",
    "RandomResizedCrop",
    "RandomFlip",
    "Resize",
    "ToTensor",
]


class TransformsCompose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img: np.ndarray, target: dict):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class RandomBrightness:
    def __init__(self, ratio: Tuple[float, float] = (0.5, 1.5)) -> None:
        self.ratio = ratio

    def __call__(
        self, img: np.ndarray, target: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            v = v.astype(np.float) * random.choice(self.ratio)
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img, target


class RandomSaturation:
    def __init__(self, ratio: Tuple[float, float] = (0.5, 1.5)) -> None:
        self.ratio = ratio

    def __call__(
        self, img: np.ndarray, target: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            s = s.astype(np.float) * random.choice(self.ratio)
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img, target


class RandomHue:
    def __init__(self, ratio: Tuple[float, float] = (0.5, 1.5)) -> None:
        self.ratio = ratio

    def __call__(
        self, img: np.ndarray, target: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            h = h.astype(np.float) * random.choice(self.ratio)
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img, target


class RandomBlur:
    def __init__(self, kerner_size: Tuple[int, int] = (5, 5)) -> None:
        self.kerner_size = kerner_size

    def __call__(
        self, img: np.ndarray, target: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        if random.random() < 0.5:
            img = cv2.blur(img, self.kerner_size)
        return img, target


class SubMean:
    def __init__(self, mean: Tuple[float, float, float] = (122.67891434, 116.66876762, 104.00698793)) -> None:
        self.mean = np.array(mean, dtype=np.float)

    def __call__(
        self, img: np.ndarray, target: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        img = img - self.mean.astype(img.dtype)
        return img, target

class ConvertColor:
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(
        self, img: np.ndarray, target: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return img, target

class RandomShift:
    def __init__(
        self, shift_scale: float = 0.6, background: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> None:
        self.shift_scale = shift_scale
        self.background = np.array(background, dtype=np.float)

    def __call__(
        self, img: np.ndarray, target: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        if random.random() < 0.5:
            h, w, c = img.shape
            center = (target[0][:, 2:] + target[0][:, :2]) / 2
            shifted_image = np.zeros((h, w, c), dtype=img.dtype)
            shifted_image[:, :] = self.background
            dx = int(random.uniform(-w * self.shift_scale, w * self.shift_scale))
            dy = int(random.uniform(-h * self.shift_scale, h * self.shift_scale))
            if dx >= 0 and dy >= 0:
                shifted_image[dy:, dx:] = img[: h - dy, : w - dx]
            elif dx >= 0 and dy < 0:
                shifted_image[: h + dy, dx:] = img[-dy:, : w - dx]
            elif dx < 0 and dy >= 0:
                shifted_image[dy:, : w + dx] = img[: h - dy, -dx:]
            elif dx < 0 and dy < 0:
                shifted_image[: h + dy, : w + dx] = img[-dy:, -dx:]
            center = center + np.broadcast_to(np.array([[dx, dy]], dtype=int), center.shape)
            mask1: np.ndarray = (center[:, 0] >= 0) & (center[:, 0] < w)
            mask2: np.ndarray = (center[:, 1] >= 0) & (center[:, 1] < h)
            mask: np.ndarray = (mask1 & mask2).reshape(-1, 1)
            boxes = target[0]
            shifted_boxes: np.ndarray = boxes[np.broadcast_to(mask, boxes.shape)].reshape(-1, 4)
            if len(shifted_boxes) == 0:
                return img, target

            box_shift = np.broadcast_to(np.array([[dx, dy, dx, dy]]), shifted_boxes.shape)
            shifted_boxes = shifted_boxes + box_shift
            shifted_boxes[:, 0] = shifted_boxes[:, 0].clip(min=0, max=w)
            shifted_boxes[:, 1] = shifted_boxes[:, 1].clip(min=0, max=h)
            shifted_boxes[:, 2] = shifted_boxes[:, 2].clip(min=0, max=w)
            shifted_boxes[:, 3] = shifted_boxes[:, 3].clip(min=0, max=h)
            target[0] = shifted_boxes
            target[1] = target[1][mask.reshape(-1)]
            img = shifted_image
        return img, target


class RandomScale:
    def __init__(self, scale: Tuple[float, float] = (0.75, 1.3)) -> None:
        self.scale = scale

    def __call__(
        self, img: np.ndarray, target: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        if random.random() < 0.5:
            scale_x = random.uniform(*self.scale)
            scale_y = random.uniform(*self.scale)
            height, width, c = img.shape
            img = cv2.resize(img, (int(width * scale_x), int(height * scale_y)))

            scale_tensor = np.array([[scale_x, scale_y, scale_x, scale_y]])
            scale_tensor = np.broadcast_to(scale_tensor, target[0].shape)
            target[0] *= scale_tensor
        return img, target


class RandomCrop:
    def __init__(self, crop_scale: float = 0.6) -> None:
        self.crop_scale = crop_scale

    def __call__(
        self, img: np.ndarray, target: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        if random.random() < 0.5:
            center = (target[0][:, 2:] + target[0][:, :2]) / 2
            height, width, _ = img.shape
            h = random.uniform(self.crop_scale * height, height)
            w = random.uniform(self.crop_scale * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - np.broadcast_to(np.array([[x, y]]), center.shape)
            mask1: np.ndarray = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2: np.ndarray = (center[:, 1] > 0) & (center[:, 1] < h)
            mask: np.ndarray = (mask1 & mask2).reshape(-1, 1)

            boxes_in: np.ndarray = target[0][np.broadcast_to(mask, target[0].shape)].reshape(-1, 4)
            if len(boxes_in) == 0:
                return img, target

            box_shift = np.broadcast_to(np.array([[x, y, x, y]]), boxes_in.shape)
            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clip(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clip(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clip(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clip(min=0, max=h)

            target[0] = boxes_in
            target[1] = target[1][mask.reshape(-1)]
            img = img[y : y + h, x : x + w, :]
        return img, target


class Resize:
    def __init__(self, img_size: Tuple[int, int]) -> None:
        self.img_size = img_size

    def __call__(
        self, img: np.ndarray, target: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        height, width, _ = img.shape
        img = cv2.resize(img, self.img_size)
        scale_tensor = np.array(
            [
                [
                    self.img_size[0] / width,
                    self.img_size[1] / height,
                    self.img_size[0] / width,
                    self.img_size[1] / height,
                ]
            ]
        )
        scale_tensor = np.broadcast_to(scale_tensor, target[0].shape)
        target[0] *= scale_tensor
        return img, target


class RandomResizedCrop:
    def __init__(self, img_size: Tuple[int, int], crop_scale: float = 0.6) -> None:
        self.crop = RandomCrop(crop_scale)
        self.resize = Resize(img_size)

    def __call__(
        self, img: np.ndarray, target: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        img, target = self.crop(img, target)
        img, target = self.resize(img, target)
        return img, target


class RandomFlip:
    def __call__(
        self, img: np.ndarray, target: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        if random.random() < 0.5:
            h, w, _ = img.shape
            img = np.fliplr(img).copy()
            xmin = w - target[0][:, 2]
            xmax = w - target[0][:, 0]
            target[0][:, 0] = xmin
            target[0][:, 2] = xmax
        return img, target


class ToTensor:
    def __call__(
        self,
        img: Union[np.ndarray, Image],
        target: Union[List[np.ndarray], np.ndarray],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if isinstance(target, list):
            return to_tensor(img), [
                torch.from_numpy(x).to(dtype=torch.get_default_dtype()) for x in target
            ]
        elif isinstance(target, tuple):
            return to_tensor(img), tuple([
                torch.from_numpy(x).to(dtype=torch.get_default_dtype()) for x in target
            ])
        elif isinstance(target, np.ndarray):
            return to_tensor(img), torch.from_numpy(target).to(dtype=torch.get_default_dtype())
        return to_tensor(img), target
