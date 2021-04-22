import json
import numpy as np
import matplotlib.pylab as plt
import cv2
from skimage.draw import polygon


from collections import defaultdict
from typing import Union, List, Tuple
from helpers import convert_from_ls


class Annotation():
    def __init__(self, filename: str):
        self.filename = filename
        self.shape = (0, 0)
        self.annotation = defaultdict(list)
        self.points = defaultdict(list)
        self.load_annotations()

    def __str__(self):
        count_values = {}
        for k, v in self.annotation.items():
            count_values[k] = len(self.annotation[k])

        return str(count_values)
        # return str(self.annotation.keys())

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
                # x = [i[0] for i in points]
                # y = [i[1] for i in points]
                points = self.points2pixel(points)
                self.points[label].append(points)
                filled_mask = self.fill_polygons(points)
                self.annotation[label].append(filled_mask)
                # x, y = self.convert2pixel(x, y)
                # mask = self.create_mask(x, y)
                # self.annotation[label] = mask
                # self.annotation[label].append(mask)

    # def convert2pixel(self, x: List, y: List) -> Tuple[int, int]:
    #     x = [round(self.shape[0] * i / 100) for i in x]
    #     y = [round(self.shape[1] * j / 100) for j in y]
    #     return x, y

    def points2pixel(self, points):
        points = [[round(self.shape[0] * p[1] / 100),
                   round(self.shape[1] * p[0] / 100)] for p in points]
        return np.array(points, dtype=np.int32)


    def crop_center(self):
        """
        Crop real image 2d numpy
        """
        y, x = self.shape
        new_x, new_y = self.cropped_shape
        startx = x//2 - (new_x // 2)
        starty = y//2 - (new_y // 2)
        print(self.gt_image.shape)
        cropped_img = self.gt_image[starty:starty+new_y, startx:startx+new_x]
        print(cropped_img.shape)
        cropped_tensor = T.to_tensor(cropped_img)
        self.cropped_kspace = fastmri.fft2c(cropped_tensor)

    # def fill_polygons(self, points: List[List]):
    #     print(points)
    #     mask = np.zeros(shape=self.shape, dtype=np.uint8)
    #     cv2.fillPoly(mask, pts=points, color=255)
    #     return mask

    def fill_polygons(self, points: List[List]):
        mask = np.zeros(shape=self.shape, dtype=np.uint8)
        r, c = polygon(points[:, 0], points[:, 1], mask.shape)
        mask[r, c] = 1
        return mask

    def create_mask(self, x, y) -> np.ndarray:
        mask = np.zeros(shape=self.shape, dtype=np.int8)
        mask[x, y] = 1
        return mask

    def get_annotation(self, label: str):
        return self.annotation[label]

    def disp_matrix(self, label: str, idx=0):
        plt.title(label)
        plt.imshow(self.annotation[label][idx], cmap='gray')
