import json
import numpy as np
import matplotlib.pylab as plt

from typing import Union, List, Tuple
from helpers import convert_from_ls


class Annotation():
    def __init__(self, filename: str):
        self.filename = filename
        self.shape = (0, 0)
        self.masks = []
        self.types = []
        self.points = []
        self.load_annotations()

    def __str__(self):
        return self.type

    def load_annotations(self):
        with open(self.filename) as f:
            segmentations = json.load(f)
            segmentations = segmentations[0]
            segmentations = segmentations["annotations"][0]
            segmentations = segmentations["result"]

            for annot in segmentations:
                self.shape = annot["original_width"], annot["original_height"]
                self.types.append(annot["value"]["polygonlabels"][0])
                points = annot["value"]["points"]
                x = [i[0] for i in points]
                y = [i[1] for i in points]
                self.points.append(points)
                x, y = convert_from_ls(np.array(x), np.array(y), shape=self.shape)
                self.set_mask(x, y)
            print(len(self.masks))

    def set_mask(self, x, y):
        mask = np.zeros(shape=self.shape, dtype=np.int8)
        mask[x, y] = 1
        self.masks.append(mask)

    def get_mask(self, mask_type: str):
        return self.mask

    def disp_matrix(self, idx):
        plt.title(self.types[idx])
        plt.imshow(self.masks[idx], cmap='gray')
