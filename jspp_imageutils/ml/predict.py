
import os
import cv2
import time
import keras
import click

from typing import Literal, Tuple, Union, Iterable

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

from jspp_imageutils.image.chunking import chunk_image_generator

# import miscellaneous modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot


LABELS_TO_NAMES = {0: "cell"}


def predict_images(model_path: str, image_dir: str):
    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')
    imagnames = os.listdir(image_dir)

    for i in imagnames:
        predict_and_plot_single_image(os.path.join(image_dir, i))

# TODO implement the function to return either data or to file

# TODO add a way to show progress
def predict_chunked_image(img_path: str, model: keras.models.Model,
                          chunk_sizes: Tuple[int, int] = (500, 500)):
    image = read_image_bgr(img_path)

    image = preprocess_image(image)

    img_gen = chunk_image_generator(
        image, chunk_size=chunk_sizes,
        displacement=chunk_sizes)
    results = []
    i = 0

    for x_start, y_start, chunk in img_gen:
        chunk, scale = resize_image(chunk)

        boxes, scores, labels = model.predict_on_batch(
            np.expand_dims(chunk, axis=0))
        
        boxes /= scale
        out_dict = {
            'x_start':  x_start,
            'y_start':  y_start,
            'scale'  :  scale,
            'boxes'  :  boxes,
            'scores' :  scores,
            'labels' :  labels
            }
        
        i += 1

        print("Done with image {i} at x:{x}, y:{y}".format(
            i = i, x = x_start,
            y = y_start))
        results.append(out_dict)

    return(results)


def plot_predictions(img: np.ndarray, 
                     scaled_boxes: Iterable[Iterable[int, int, int, int]],
                     scores: Iterable[float],
                     labels: Iterable[int],
                     label_dict: dict,
                     out_figsize: Tuple[int, int],
                     score_cutoff: float = 0.) -> pyplot:

    draw = img.copy()
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
                                  output: Union[Literal["print"], str] = ""):
    # load image
    image = read_image_bgr(img_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # Plot the image without boxes
    pyplot.figure(figsize=(10, 10))
    pyplot.axis('off')
    pyplot.imshow(draw)

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


def entry_to_df(entry):
    COLUMN_NAMES = [
        "section_x_start",
        "section_y_start",
        "box_x1", "box_y1",
        "box_x2", "box_y2",
        "score", "label"]

    iter = zip(entry['boxes'][0],
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

    df = pd.DataFrame(pd_list, columns = COLUMN_NAMES)
    return(df)


def results_to_df(res):
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
        2. Output the prediction results to the chosen directory (add option to force over-write)
            - Output as csv
            - Output as predicted Image 
                - TODO [add option for multi-class prediction for color coding]
    """
    pass

if __name__ == "__main__":
    cli()