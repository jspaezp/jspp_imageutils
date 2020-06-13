import itertools
import numpy as np

from nptyping import NDArray, UInt
from jspp_imageutils.image.types import GenImgArray, GenImgBatch

from typing import Tuple, Iterable, Iterator, Any, Union
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# TODO: fix everywhere the x and y axis nomenclature
"""
chunk_image_on_position -> returns images
chunk_image_generator -> returns images
chunk_data_image_generator -> returns batches of data
"""


def chunk_image_on_position(arr_img: GenImgArray,
                            x_pos: Iterable[int], y_pos: Iterable[int],
                            dimensions: Tuple[int, int] = (50, 50),
                            warn_leftovers=True) -> Iterator[Tuple[int, int, GenImgArray]]:

    # TODO decide if this should handle centering the points ...
    x_ends = [x + dimensions[0] for x in x_pos]
    y_ends = [y + dimensions[1] for y in y_pos]

    i = 0
    # TODO find a better way to indent this ...
    for y_start, y_end, x_start, x_end in \
            zip(y_pos, y_ends, x_pos, x_ends):
        temp_arr_img = arr_img[x_start:x_end, y_start:y_end, ]
        if temp_arr_img.shape[0:2] == dimensions:
            yield x_start, y_start, temp_arr_img
            i += 1
        else:
            if warn_leftovers:
                print("skipping chunk due to weird size",
                      str(temp_arr_img.shape))

    print("Image generator yielded ", str(i), " images")


def chunk_image_generator(img,
                          chunk_size: Tuple[int, int] = (500, 500),
                          displacement: Tuple[int, int] = (250, 250),
                          warn_leftovers=True) -> Iterator[Tuple[int, int, GenImgArray]]:
    """
    Gets an image read with tensorflow.keras.preprocessing.image.load_img
    and returns a generator that iterates over rectangular areas of it.

    chunks are of dims (chunk_size, colors)
    """
    # TODO unify the input for this guy ...
    arr_img = np.asarray(img)
    dims = arr_img.shape

    x_starts = [
        displacement[0] * x for x in range(dims[0] // displacement[0])
    ]
    x_starts = [x for x in x_starts if
                x >= 0 & (x + chunk_size[0]) < dims[0]]
    y_starts = [
        displacement[1] * y for y in range(dims[1] // displacement[1])
    ]
    y_starts = [y for y in y_starts if
                y >= 0 & (y + chunk_size[1]) < dims[1]]

    coord_pairs = itertools.product(x_starts, y_starts)
    coord_pairs = np.array(list(coord_pairs))

    my_gen = chunk_image_on_position(
        arr_img, coord_pairs[:, 0], coord_pairs[:, 1],
        dimensions=chunk_size, warn_leftovers=warn_leftovers)

    for chunk in my_gen:
        yield(chunk)


def chunk_data_image_generator(img: GenImgArray,
                               chunk_size: Tuple[int, int] = (500, 500),
                               displacement: Tuple[int, int] = (250, 250),
                               batch: int = 16) -> GenImgBatch:
    """
    Gets an image read with tensorflow.keras.preprocessing.image.load_img
    and returns a generator that iterates over BATCHES of rectangular
    areas of it

    dimensions are (batch, chunk_size, colors)
    """
    # np.concatenate((a1, a2))
    img_generator = chunk_image_generator(
        img=img, chunk_size=chunk_size,
        displacement=displacement)

    counter = 0
    img_buffer = []

    for _, _, temp_arr_img in img_generator:
        tmp_arr_dims = temp_arr_img.shape
        temp_arr_img = temp_arr_img.reshape(1, *tmp_arr_dims)
        img_buffer.append(temp_arr_img)
        counter += 1

        if counter == batch:
            yield(np.concatenate(img_buffer))

            counter = 0
            img_buffer = []

    yield(np.concatenate(img_buffer))

