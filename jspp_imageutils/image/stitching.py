
import numpy as np
from typing import Tuple, Iterable


def stitch_images(chunk_gen: Iterable[np.array],
                  img_shape: Tuple[int, int],
                  displacement: Tuple[int, int] = (250, 250),
                  operation="replace") -> np.array:
    """
    Given an iterator or arrays, should return a larger array/image that 
    combines all the other ones.

    Should in theory be the opposite function to "chunk_image_generator"

    hypothetically,

    >>> gen = chunk_image_generator(img)
    >>> img2 = stitch_images(gen)

    img and img2 should be the same
    """

    img = np.zeros(img_shape)

    x_start_pos = 0
    y_start_pos = 0

    for _, _, chunk in chunk_gen:

        chunk_dims = chunk.shape

        x_end_pos = x_start_pos + chunk_dims[0]
        y_end_pos = y_start_pos + chunk_dims[1]

        if x_end_pos > (img_shape[0]):
            x_start_pos = 0
            x_end_pos = x_start_pos + chunk_dims[0]

            y_start_pos = y_start_pos + displacement[1]
            y_end_pos = y_start_pos + chunk_dims[1]

        img_slice = img[y_start_pos:y_end_pos, x_start_pos:x_end_pos]

        if operation == "replace":
            replacement_slice = chunk.reshape(img_slice.shape)
        elif operation == "add":
            replacement_slice = img_slice + chunk.reshape(img_slice.shape)
        else:
            raise ValueError(
                "Non supported operation for replacement,",
                "please use either 'add' or 'replace'")

        # TODO implement operations to handle overlaps
        img_slice[:, :] = replacement_slice

        x_start_pos = x_start_pos + displacement[0]

    return(img)
