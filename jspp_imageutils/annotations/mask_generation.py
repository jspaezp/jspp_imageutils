
import cv2
import numpy as np
import scipy.stats as st

from typing import Iterable, Tuple, Union
from PIL import Image


def gauss_kern(kernlen: int = 5, nsig: float = 1):
    """
    Returns a 2D Gaussian kernel. Answer to question 29731726 in stack overflow
    """
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


def generate_density_dist(coords: Iterable[Tuple[int, int]],
                          img_dim: Tuple[int, int],
                          kernlen: int = 5, nsig: float = 1.0,
                          flip=None, transpose=None):

    img = np.zeros(img_dim)
    kern_add = gauss_kern(kernlen=kernlen, nsig=nsig)
    kern_add = kern_add / np.max(kern_add)

    for x_coords, y_coords in coords:
        # TODO add a way to handle going around the edges ...
        kern_x_start = x_coords - (kern_add.shape[0] // 2)
        kern_x_end = kern_x_start + kern_add.shape[0]

        kern_y_start = y_coords - (kern_add.shape[1] // 2)
        kern_y_end = kern_y_start + kern_add.shape[1]

        try:
            img_slice = img[kern_x_start:kern_x_end,
                            kern_y_start:kern_y_end]
            img_slice[0:] = img_slice + kern_add
        except ValueError:
            # This just ignores all instances where the kernell
            #  would be outside of the area ...
            pass

    if flip is not None:
        img = np.flip(img, flip)

    if transpose is not None:
        img = np.transpose(img, transpose)

    return(img)


def generate_discrete_density_dist(coords: Iterable[Tuple[int, int]],
                                   img_dim: Tuple[int, int],
                                   radius: Union[float, int],
                                   max_val: int = 2,
                                   flip=None, transpose=None):

    array = np.zeros(img_dim)

    for a, b in coords:

        y, x = np.ogrid[-a:img_dim[0]-a, -b:img_dim[1]-b]
        mask = x*x + y*y <= radius*radius

        array[mask] = array[mask] + 1

    if flip is not None:
        array = np.flip(array, flip)

    if transpose is not None:
        array = np.transpose(array, transpose)

    return(array)


def save_density_map_as_binary_img(arr, filename, cutoff=0.1):
    arr = (arr > cutoff).astype('uint8')
    cv2.imwrite(filename, arr)
    print("File saved for ", filename)


def density_map_as_img(arr, img_dim: Tuple[int, int],
                       scale: bool = True, norm: bool = True,
                       inv: bool = True):
    # TODO evaluate if having the img_dim argument is necessary
    if norm:
        arr = arr/np.max(arr)

    if inv:
        # TODO add a check to check if it shoudl be subtracted from 255 or 1
        arr = 1.0-arr

    if scale:
        arr = arr*255.0

    norm_arr = arr.astype(np.uint8)
    norm_arr.resize(img_dim)
    # Mode "L" means 8 bit grayscale
    # for more details refer to ...
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    img = Image.fromarray(norm_arr, mode="L")
    return(img)


if __name__ == "__main__":
    coords = ((25, 25), (75, 75))
    img_dim = (300, 300)

    arr = generate_density_dist(coords, img_dim, kernlen=25, nsig=3)
    img = density_map_as_img(arr, img_dim=img_dim)

    img.save('my.png')

    # Known issue ...
    coords = ((1, 1), (75, 75))
    img_dim = (300, 300)

    arr = generate_density_dist(coords, img_dim, kernlen=25, nsig=3)
    img = density_map_as_img(arr, img_dim=img_dim)

    img.save('my2.png')
