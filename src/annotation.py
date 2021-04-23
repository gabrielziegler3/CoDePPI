import json
import numpy as np
import matplotlib.pylab as plt

from skimage.draw import polygon
from collections import defaultdict
from typing import Union, List, Tuple


class Annotation():
    def __init__(self, filename: str, crop_shape: Tuple[int, int] = None):
        self.filename = filename
        self.crop_shape = crop_shape
        self.shape = (0, 0)
        self.annotation = defaultdict(list)
        self.cropped_annotation = defaultdict(list)
        self.points = defaultdict(list)
        self.load_annotations()

        if self.crop_shape:
            self.crop_center(self.crop_shape)

    def __str__(self) -> str:
        count_values = {}
        for k, v in self.annotation.items():
            count_values[k] = len(self.annotation[k])

        return str(count_values)

    def load_annotations(self):
        with open(self.filename) as f:
            segmentations = json.load(f)
            segmentations = segmentations[0]
            segmentations = segmentations["annotations"][0]
            segmentations = segmentations["result"]

            for annot in segmentations:
                self.shape = annot["original_height"], annot["original_width"]
                label = annot["value"]["polygonlabels"][0]
                points = annot["value"]["points"]
                points = self.points2pixel(points)
                self.points[label].append(points)
                filled_mask = self.fill_polygons(points)
                self.annotation[label].append(filled_mask)

    def points2pixel(self, points):
        points = [[round(self.shape[0] * p[1] / 100),
                   round(self.shape[1] * p[0] / 100)] for p in points]
        return np.array(points, dtype=np.int32)

    def crop_center(self, crop_shape: Tuple[int, int]):
        y, x = self.shape
        new_x, new_y = crop_shape
        startx = x//2 - (new_x // 2)
        starty = y//2 - (new_y // 2)
        cropped_annotation = defaultdict(list)

        for k, v in self.annotation.items():
            for idx, label_annot in enumerate(self.annotation[k]):
                cropped_annotation[k].append(label_annot[starty:starty+new_y, startx:startx+new_x])

        self.shape = crop_shape
        self.annotation = cropped_annotation

    def fill_polygons(self, points: List[List]) -> np.ndarray:
        mask = np.zeros(shape=self.shape, dtype=np.uint8)
        r, c = polygon(points[:, 0], points[:, 1], mask.shape)
        mask[r, c] = 1
        return mask

    def create_mask(self, x, y) -> np.ndarray:
        mask = np.zeros(shape=self.shape, dtype=np.int8)
        mask[x, y] = 1
        return mask

    def get_phi(self,) -> dict:
        phi = {}
        for k, v in self.annotation.items():
            phi[k] = [np.sort(np.nonzero(annot.flatten())[0]) for annot in v]

        return phi

    def get_annotations(self) -> dict:
        return self.annotation

    def get_annotation_by_label(self, label: str) -> np.ndarray:
        return self.annotation[label]

    def disp_matrix(self, label: str, idx=0):
        if label == "all":
            masks = np.zeros(shape=self.shape, dtype=np.uint8)
            for annot in list(self.annotation.values()):
                for label_annot in annot:
                    masks += label_annot
            plt.imshow(masks, cmap='gray')
            plt.colorbar()
        else:
            plt.imshow(self.annotation[label][idx], cmap='gray')
