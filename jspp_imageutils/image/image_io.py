
import os
import cv2
from tensorflow.keras.preprocessing.image import save_img


def save_generator_images(chunk_gen, output_dir: str,
                          basename: str, extension: str):

    try:
        os.makedirs(output_dir)
    except FileExistsError:
        # directory already exists
        pass

    for i, chunk in enumerate(chunk_gen):
        filename = basename + "_" + str(i) + extension
        out_path = os.path.join(output_dir, filename)

        cv2.imwrite(out_path, chunk)

    print("Saved ", str(i+1), " images to " + output_dir)
