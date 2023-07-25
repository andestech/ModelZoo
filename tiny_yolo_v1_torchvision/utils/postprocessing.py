from typing import List, Tuple

import numpy as np

__all__ = ["YOLOv1TrainPostProcessing"]


class YOLOv1TrainPostProcessing:
    def __init__(
        self,
        num_grids: int = 7,
        num_bboxes: int = 2,
        num_classes: int = 20,
    ) -> None:
        self.num_grids = num_grids
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes

    def __call__(self, img: np.ndarray, target: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        img_height, img_width, _ = img.shape

        label = np.zeros((self.num_grids, self.num_grids, 5 * self.num_bboxes + self.num_classes))
        for box, category in zip(target[0], target[1]):
            xc = (box[0] + box[2]) / 2 / img_width
            yc = (box[1] + box[3]) / 2 / img_height
            w = (box[2] - box[0]) / img_width
            h = (box[3] - box[1]) / img_height
            idx_x = int(xc * self.num_grids)
            idx_y = int(yc * self.num_grids)
            delta_x = xc * self.num_grids - idx_x
            delta_y = yc * self.num_grids - idx_y

            for i in range(self.num_bboxes):
                label[idx_y][idx_x][5 * i] = delta_x
                label[idx_y][idx_x][5 * i + 1] = delta_y
                label[idx_y][idx_x][5 * i + 2] = np.sqrt(w)
                label[idx_y][idx_x][5 * i + 3] = np.sqrt(h)
                label[idx_y][idx_x][5 * i + 4] = 1
            label[idx_y][idx_x][self.num_bboxes * 5 + category] = 1
        return img, label
