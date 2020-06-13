#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
It intends to augment data with bounding boxes.

The data is assumed to be large and will generate small subsets of the
large immage with their corresponding bounding boxed

reads bounding boxes from a csv file and outputs
1. new augmented image
2. new csv file with the corresponding bounding boxes
3. new file with the image and drawn boxes (image)

Usage:
    $ python3 augment_boxed_large.py --in_img ../img/118649_A_5_7.jpg \
        --in_csv ../box_annotations/118649_A_5_7.csv \
        --out_img_dir ../augmented/img/ \
        --out_mask_dir ../augmented/mask/ \
        --out_csv_dir ../augmented/csv/ \
        --height 512 \
        --width 512 \
        --seed 1 \
        --num_generate 200

"""

import cv2
import os
import argparse

import pandas as pd
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap

from typing import Tuple

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


GREEN = [0, 255, 0]
ORANGE = [255, 140, 0]
RED = [255, 0, 0]


CSV_COLUMN_NAMES = [
    'filename',
    'xmin',
    'ymin',
    'xmax',
    'ymax',
    'class']


# Draw BBs on an image
# and before doing that, extend the image plane by BORDER pixels.
# Mark BBs inside the image plane with green color, those partially inside
# with orange and those fully outside with red.
def draw_bbs(img, bbs: BoundingBox):
    image_border = img
    for box in bbs.bounding_boxes:
        if box.is_fully_within_image(img.shape):
            color = GREEN
        elif box.is_partly_within_image(img.shape):
            color = ORANGE
        else:
            color = RED
        image_border = box.draw_on_image(image_border, size=2, color=color)

    return(image_border)


def generate_sequential_augmenter(width: int = 512, height: int = 512) -> iaa.Sequential:
    seq = iaa.Sequential([
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Apply affine transformations to each image.
        # Scale/zoom them, otate them and shear them.
        iaa.Sometimes(
            0.5,
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                rotate=(-25, 25),
                shear=(-8, 8))
        ),
        # Crops to a given size, uniformly and skipping
        # 10% of the image in all edges
        iaa.size.CropToFixedSize(
            width, height,
            position=(iap.Uniform(0.1, 0.9), iap.Uniform(0.1, 0.9))),
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Flipud(0.5),  # vertical flips
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.85, 1.15)),
        # Make some images brighter and some darker.
        # In 10% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.9, 1.1), per_channel=0.2)
    ], random_order=False)  # apply augmenters in the explicit order
    return(seq)


def bbs_from_csv(csv_file: str, img_shape: Tuple) -> BoundingBoxesOnImage:
    box_df = pd.read_csv(csv_file)

    # Generate the bounding box objects from the csv data inputs
    bbs_list = []

    position_iterator = zip(
        box_df["xmin"], box_df["xmax"],
        box_df["ymin"], box_df["ymax"],
        box_df["class"])

    for x1, x2, y1, y2, label in position_iterator:
        bbs_list.append(BoundingBox(
            x1=x1, x2=x2,
            y1=y1, y2=y2,
            label=label))

    bbs = BoundingBoxesOnImage(bbs_list, shape=img_shape)
    return(bbs)


def main(in_img: str, in_csv: str, out_img_dir: str, out_csv_dir: str,
         out_mask_dir: str, num_generate: int = 200,
         height: int = 512, width: int = 512):
    # Reads the input data
    img = cv2.imread(in_img)

    # Generates the sequential augmentator
    seq = generate_sequential_augmenter(width=width, height=height)

    # Generate the base names used to save the new files
    # basename is the name without the extension
    base = os.path.basename(in_img)
    basename = os.path.splitext(base)[0]

    # Convert output dir of the image to an absolute path
    # (it will be used in the csv annotation file)
    out_img_dir = os.path.abspath(out_img_dir)

    # Generate the bounding boxes
    bbs = bbs_from_csv(in_csv, img.shape)

    # Actually generate the augmentations
    for i in range(num_generate):
        # For every augmentation we generate new file
        # names (image, file and csv)
        new_img_name = 'aug_'+str(i)+'_' + basename + '.png'
        new_img_name = os.path.join(out_img_dir, new_img_name)

        csv_img_name = 'csv_'+str(i)+'_' + basename + '.csv'
        csv_img_name = os.path.join(out_csv_dir, csv_img_name)

        mask_img_name = 'masked_'+str(i)+'_' + basename + '.png'
        mask_img_name = os.path.join(out_mask_dir, mask_img_name)

        # Generate single augmentation image and boxes
        image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)

        # Crop boxes to be inside the image and
        # remove the ones that are more than 90% outside of it
        bbs_out = bbs_aug.remove_out_of_image_fraction(0.9).clip_out_of_image()

        if len(bbs_out) > 0:
            # Generate masked images
            masked_image = draw_bbs(image_aug, bbs_out)
        else:
            masked_image = image_aug

        # Tranform boxes into csv outputs
        # The columns are organized as the csv required by keras-retinanet
        # https://github.com/fizyr/keras-retinanet#csv-datasets
        # path/to/image.jpg,x1,y1,x2,y2,class_name
        csv_values_list = []

        for bb in bbs_out:
            # Remove bounding boxes less than 2 pixels wide
            if abs(bb.x1 - bb.x2) < 2 or abs(bb.y1 - bb.y2) < 2:
                continue
            else:
                value = (
                    new_img_name,
                    bb.x1, bb.y1,
                    bb.x2, bb.y2,
                    bb.label)
                csv_values_list.append(value)

        if len(csv_values_list) == 0:
            csv_values_list.append((
                new_img_name,
                None, None,
                None, None,
                None))

        csv_df = pd.DataFrame(csv_values_list, columns=CSV_COLUMN_NAMES)

        # Save csv
        csv_df.to_csv(csv_img_name, index=None)

        # Save image
        cv2.imwrite(new_img_name, image_aug)

        # Save masked image
        cv2.imwrite(mask_img_name, masked_image)


def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_img", type=str)
    parser.add_argument("--in_csv", type=str)
    parser.add_argument("--out_img_dir", type=str)
    parser.add_argument("--out_csv_dir", type=str)
    parser.add_argument("--out_mask_dir", type=str)
    parser.add_argument("--num_generate", type=int, default=100)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    return(args)


if __name__ == '__main__':
    args = parse()
    ia.seed(args.seed)
    main(in_img=args.in_img,
         in_csv=args.in_csv,
         out_img_dir=args.out_img_dir,
         out_csv_dir=args.out_csv_dir,
         out_mask_dir=args.out_mask_dir,
         num_generate=args.num_generate,
         height=args.height,
         width=args.width)
