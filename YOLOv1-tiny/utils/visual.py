from typing import List, Union

import numpy as np
from matplotlib import cm, patches
from matplotlib import pyplot as plt
from PIL.Image import Image

__all__ = ["plot_boxes"]


def plot_boxes(
    img: Union[np.ndarray, Image],
    boxes: np.ndarray,
    labels: np.ndarray = None,
    class_names: Union[None, List[str]] = None,
    color: Union[None, tuple] = None,
    file_name: Union[None, str] = None,
    show: bool = True,
) -> None:
    """
    Args:
        img (np.ndarray): display image, sized [width, height, 3]
        boxed (np.ndarray): detected bounding boxes which size is [n_bboxes, 4], 4=[x_center, y_center, w, h]
        labels (np.ndarray): label of bounding boxes which size is [n_bboxes]
        class_names (list[str]): all the class names
        color (tuple): color of bounding boxes
        file_name (str): if you want to save image, setting the file name
    """
    assert (
        len(img.shape) == 3 and img.shape[-1] == 3
    ), f"Expected image shape is [width, height, 3], got {img.shape}"
    assert isinstance(img, np.ndarray) or isinstance(
        img, Image
    ), f"Expected image is {np.ndarray} or {Image}, got {type(img)}"
    assert isinstance(boxes, np.ndarray), f"Expected img is {type(np.ndarray)}, got {type(boxes)}"
    if labels is not None:
        if len(labels.shape) == 2:
            labels = np.argmax(labels, axis=-1)
        assert len(labels.shape) == 1, f"Expected image shape is [n_bboxes1], got {labels.shape}"
        assert isinstance(
            labels, np.ndarray
        ), f"Expected img is {type(np.ndarray)}, got {type(labels)}"
    if color == None:
        color_map = cm.get_cmap("hsv")
        color = color_map(np.random.rand())

    _, ax = plt.subplots()
    ax.imshow(img)
    width = img.shape[1]
    height = img.shape[0]
    for i, box in enumerate(boxes):
        x = (box[0] - box[2] / 2) * width
        y = (box[1] - box[3] / 2) * height
        w = box[2] * width
        h = box[3] * height

        if labels.any() != None and class_names != None:
            color = color_map(labels[i] / len(class_names))
            ax.text(x, y, class_names[int(labels[i])], color=color)
        rect = patches.Rectangle((x, y), w, h, edgecolor=color, linewidth=2, fill=False)
        ax.add_patch(rect)

    if file_name != None:
        plt.savefig(file_name)
    if show:
        plt.show()


if __name__ == "__main__":
    from datasets.voc import VOC_CLASS, voc_detection

    dataloader = voc_detection()
    for x, y in dataloader["val"]:
        img = x[0].permute(1, 2, 0)
        boxes = y[0][:, :, :4]
        boxes = boxes[boxes.nonzero(as_tuple=True)].view(-1, 4)
        labels = y[0][:, :, 10::]
        labels = labels.nonzero(as_tuple=True)[2]
        img = plot_boxes(img.numpy(), boxes.numpy(), labels=labels.numpy(), class_names=VOC_CLASS)
