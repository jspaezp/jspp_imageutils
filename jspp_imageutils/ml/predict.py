
import os
import cv2
import time
import keras
import click

from typing import Literal, Tuple, Union, Iterable, Dict

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image
from keras_retinanet.utils.image import resize_image
from keras_retinanet.utils.visualization import draw_box
from keras_retinanet.utils.colors import label_color

from jspp_imageutils.image.chunking import chunk_image_generator

# import miscellaneous modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot


LABELS_TO_NAMES = {0: "cell"}


# TODO implement the function to return either data or to file
def predict_images(model_path: str, image_dir: str,
                   backbone_name: str = 'resnet50'):
    """
    predict_images using a model trained with keras-retinanet

    Predicts all images within a directory, using a detection model
    trained with keras-retinanet.

    :param model_path: path to the file where the model is stored
        (.h5 file)
    :type model_path: str
    :param image_dir: path to the directory containing all the images
    :type image_dir: str
    :param backbone_name: model backbone used in the retinanet,
        defaults to 'resnet50'
    :type backbone_name: str, optional
    """

    # load retinanet model
    model = models.load_model(model_path, backbone_name=backbone_name)
    imagnames = os.listdir(image_dir)

    for i in imagnames:
        predict_and_plot_single_image(os.path.join(image_dir, i), model)


# TODO add a way to show progress
def predict_chunked_image(img_path: str, model: keras.models.Model,
                          chunk_sizes: Tuple[int, int] = (500, 500)):

    print("Reading Image")
    image = read_image_bgr(img_path)

    print("Processing Image")
    image = preprocess_image(image)

    img_gen = chunk_image_generator(
        image, chunk_size=chunk_sizes,
        displacement=chunk_sizes)
    results = []
    i = 0

    print("Predicting")
    for x_start, y_start, chunk in img_gen:
        chunk, scale = resize_image(chunk)

        boxes, scores, labels = model.predict_on_batch(
            np.expand_dims(chunk, axis=0))

        boxes /= scale
        out_dict = {
            'x_start':  x_start,
            'y_start':  y_start,
            'scale':    scale,
            'boxes':    boxes,
            'scores':   scores,
            'labels':   labels}

        i += 1

        print("Done with image {i} at x:{x}, y:{y}".format(
            i=i, x=x_start,
            y=y_start))
        results.append(out_dict)

    return(results)


def plot_predictions(img: np.ndarray,
                     scaled_boxes: Iterable[Tuple[int, int, int, int]],
                     scores: Iterable[float],
                     labels: Iterable[int],
                     label_dict: Dict,
                     out_figsize: Tuple[int, int],
                     score_cutoff: float = 0.) -> pyplot:

    draw = img.copy()
    # TODO check if it is always necessary to convert the color mode ...
    # is it asuming cv2 or PIL input format??

    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    # TODO handle instances when scores has a single element
    for box, score, label in zip(scaled_boxes, scores, labels):
        # scores are sorted so we can break
        if score < score_cutoff:
            continue

        color = label_color(int(score // 0.1))

        b = np.array(box).astype(int)
        draw_box(draw, b, color=color)

        # caption = "{} {:.3f}".format(label_dict[label], score)
        # draw_caption(draw, b, caption)

    pyplot.figure(figsize=out_figsize)
    pyplot.axis('off')
    pyplot.imshow(draw)

    return(pyplot)


def predict_and_plot_single_image(img_path: str,
                                  model: keras.models.Model,
                                  output: Union[Literal["print"], str] = "") \
                                  -> None:
    """
    predict_and_plot_single_image from file and displays the figure

    takes a file path and a keras-retinanet loaded model, predicts the
    instances in the image and displays the results

    :param img_path: file path of the image to use for prediction
    :type img_path: str
    :param model: Keras retinanet loaded model
    :type model: keras.models.Model
    :param output: Can be either literally "print" or a file path to where the
                   figure will be saved, defaults to ""
    :type output: Union[Literal[, optional
    """
    # Output is displaying the image as a side effect

    # load image
    image = read_image_bgr(img_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # Plot the image without boxes
    pyplot.figure(figsize=(10, 10))
    pyplot.axis('off')
    pyplot.imshow(draw)

    # TODO make it such that the figures are shown side by side and
    # TODO add argument for figure size
    if output == "print":
        pyplot.show(output)
    else:
        pyplot.savefig(output)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(
        np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale
    print(scores)
    # visualize detections
    plt = plot_predictions(
        img=draw, boxes=boxes, scores=scores,
        labels=labels, label_dict=LABELS_TO_NAMES)
    plt.show()


def entry_to_df(entry: Dict[str, Iterable[np.ndarray]]) -> pd.DataFrame:
    """
    entry_to_df converts the output of a keras-retinanet model to a data frame

    keras retinanet models return a dictionary with multiple elements,
    this function converts that dictionary to a pandas DataFrame,
    each bounding box as a row.

    :param entry: a predction output (dictionary with boxes, scores and labels)
    :type entry: Dict[str, Iterable[np.ndarray]]
    :return: a DataFrame with a bounding box per row
    :rtype: pd.DataFrame
    """
    COLUMN_NAMES = [
        "section_x_start",
        "section_y_start",
        "box_x1", "box_y1",
        "box_x2", "box_y2",
        "score", "label"]

    # TODO generalize this function
    # TODO since it could easily work with batched inputs
    iter = zip(
        entry['boxes'][0],
        entry['scores'][0],
        entry['labels'][0])

    pd_list = []
    for box, score, label in iter:
        if score < 0.0:
            continue

        pd_list.append(
            [entry['x_start'],
             entry['y_start'],
             box[0], box[1],
             box[2], box[3],
             score, label])

    df = pd.DataFrame(pd_list, columns=COLUMN_NAMES)
    return(df)


def results_to_df(res: Iterable[Dict[int, str]]) -> pd.DataFrame:
    df_list = []

    for e in res:
        tmp_df = entry_to_df(e)
        df_list.append(tmp_df)

    full_df = pd.concat(df_list)
    return(full_df)


@click.group()
def cli():
    """
    Should hypothetically work as...

    'command call' --model foo.h5 --img imagx.jpeg imgy.jpeg \
        --out_dir some_path --backbone (optional)resnet50 \
        --force (optional) \
        --color_score/--color_class

    1. load the model
    2. for each image
        0. Assert that the image is a reasonable size and chunk if needed
        1. use the model to predict the images
        2. Output the prediction results to the chosen directory
           (add option to force over-write)
            - Output as csv
            - Output as predicted Image
                - TODO [add option for multi-class prediction for color coding]
    """
    raise NotImplementedError


if __name__ == "__main__":
    cli()
