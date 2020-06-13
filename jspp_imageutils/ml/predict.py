
import os
import cv2
import time
import keras

from typing import Literal, Tuple

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

from jspp_imageutils.image.chunking import chunk_image_generator

# import miscellaneous modules
import numpy as np
import matplotlib.pyplot as pyplot


LABELS_TO_NAMES = {0: "cell"}


def predict_images(model_path: str, image_dir: str):
    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')
    imagnames = os.listdir(image_dir)

    for i in imagnames:
        predict_and_plot_single_image(os.path.join(image_dir, i))

# TODO implement the function to return either data or to file


def predict_chunked_image(img_path, model,
                          chunk_sizes:Tuple = (500, 500)):
    image = read_image_bgr(img_path)

    image = preprocess_image(image)

    img_gen = chunk_image_generator(image, chunk_size=(500, 500), displacement=(500,500))
    results = []

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

        results.append(out_dict)
    return results


def plot_predictions(img, scaled_boxes, scores,
                     labels, label_dict) -> pyplot:

    draw = img.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    # TODO handle instances when scores has a single element
    for box, score, label in zip(scaled_boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.1:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(label_dict[label], score)
        draw_caption(draw, b, caption)

    pyplot.figure(figsize=(10, 10))
    pyplot.axis('off')
    pyplot.imshow(draw)

    return(pyplot)



def predict_and_plot_single_image(img_path: str,
                                  model: keras.models.Model,
                                  output: Literal["file", "print"]):
    # load image
    image = read_image_bgr(img_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # Plot the image without boxes
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()

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
