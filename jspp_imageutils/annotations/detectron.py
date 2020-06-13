

import os
import cv2 as cv2  # hack that helps the linter to recognize the submodules
import numpy as np
import json
import random

from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

from typing import TypedDict, List, Union, Literal

from matplotlib import pyplot as plt


class BoxList(TypedDict):
    bbox: [float, float, float, float]
    bbox_mode: Union[Literal[BoxMode.XYXY_ABS],  Literal[BoxMode.XYWH_ABS]]
    segmentation: List[float, ...]
    category_id: int


class Record(TypedDict):
    file_name: str
    file_id: Union[int, str]
    height: float
    width: float
    annotations: Union[List[BoxList, ...], List[()]]


DetectronDataset = List[Record]


# This function expects the directories to be absolute ...
# or does it ...
# Currently only supports single class annotation datasets
# TODO add support for multiple annotation datasets, could
#      use the class annotation csv
def get_record(img_file: str, csv_file: str, file_id: str) -> Record:
    record = {}

    height, width = cv2.imread(img_file).shape[:2]

    record["file_name"] = img_file
    record["file_id"] = file_id
    record["height"] = height
    record["width"] = width

    # The annotation requires a polygon
    # as [x1,y1, x2. y3 ... xn, yn]
    objs = []

    with open(csv_file, "r") as f:
        # Since a square has only 4 points, that can be
        # delimited as xmax, xmin, ymax, ymin
        # The box can be defined as  [xmax, ymax, xmin, ymax, xmin, ymin, xmax, ymin, xmax, ymax]
        next(f)  # first line should be just names
        for i in f:
            # TODO assert that the file name in the csv matches the file name in the
            #      image
            line_data = i.strip().split(",")

            if all([x == "" for x in line_data[1:-1]]):
                # objs.append({})
                continue

            # path/to/image.jpg,x1,y1,x2,y2,class_name
            # TODO implement a way to have multiple annotations
            xmin = float(line_data[1])
            ymin = float(line_data[2])
            xmax = float(line_data[3])
            ymax = float(line_data[4])

            poly = [xmax, ymax, xmin, ymax,
                    xmin, ymin, xmax, ymin,
                    xmax, ymax]
            poly = [x + 0.5 for x in poly]
            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0}
            objs.append(obj)
    record["annotations"] = objs
    return(record)


def get_dataset(dir: str) -> DetectronDataset:
    """Get datasets to load in detection

    Args:
        dir (str): A string pointing to a drectory that contains
                   an img and a csv subdirectory, where the 
                   corresponding annotations are located.


    Returns:
        DetectronDataset: a list of dictionries compatible with detectron
    """

    img_dir = os.path.join(dir, "img")
    csv_dir = os.path.join(dir, "csv")
    files = os.listdir(img_dir)
    file_basenames = [os.path.splitext(x)[0] for x in files]

    dataset_dicts = []

    for i, f in enumerate(file_basenames):
        record = get_record(
            os.path.join(img_dir, f + ".png"),
            os.path.join(csv_dir, f.replace("aug_", "csv_") + ".csv"),
            i
        )
        if len(record["annotations"]) > 0:
            dataset_dicts.append(record)

    return(dataset_dicts)


def visualize_detectron_dataset(data_dir, metadata_catalog_name: str):
    train_dicts = get_dataset(data_dir)
    
    dataset_metadata = MetadataCatalog.get("cells_train")
    for d in random.sample(train_dicts, 3):
        print(d)
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(
            img[:, :, ::-1], metadata=dataset_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        # TODO replace cv2_imshow, it is an import from google collab
        # cv2_imshow(vis.get_image()[:, :, ::-1])
        img2 = vis.get_image()[:, :, ::-1]
        plt.imshow(img2)
