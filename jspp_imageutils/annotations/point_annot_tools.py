import random
import numpy as np

from typing import Tuple, Iterable, Any, Union
from nptyping import NDArray

from jspp_imageutils.image.chunking import chunk_image_on_position


NumArray2D = Union[NDArray[(Any, Any), float], NDArray[(Any, Any),int]]


def nn_distance(X: NumArray2D, Y: NumArray2D, sqrt: bool=False):
    """
    Stack overflow question number 52366421
    Calculates the nearest neighbor distances between all elements in
    X and Y of shapes (i, n) and (j, n)

    The distance D with shape (i,j)
    between every point in X and every point in Y
    """
    D_sq = np.einsum('ij, ij ->i', X, X)[:, None] + \
        np.einsum('ij, ij ->i', Y, Y) - 2 * X.dot(Y.T)

    if (sqrt):
        D = np.sqrt(D_sq)
        return(D)
    else:
        return(D_sq)


def radius_search(X: NumArray2D, Y: NumArray2D, radius: float=1.):
    """
    Stack overflow question number 52366421
    For X and Y of shapes (i, n) and (j, n)
    Returns the indices r_i, r_j and distance r_d of every point in X 
    within distance r of every point j in Y

    Therefore, the indices in r_i are the points in X that do have a near
    neighbor in the Y array
    """
    D_sq = nn_distance(X, Y, sqrt=False)

    mask = D_sq < radius**2
    r_i, r_j = np.where(mask)
    r_d = np.sqrt(D_sq[mask])
    return(r_i, r_j, r_d)


def generate_negative_data_point_chunks(
        img_arr, coords_x: Iterable, coords_y: Iterable,
        dimensions: Tuple[int, int], num_data: int = 1000,
        num_data_try: int = 10000, mindistance: float = 30.,
        edge_dist_avoid: int = 100):

    img_dims = img_arr.shape
    # img_copy = img_arr
    # TODO implement a way to see the removed rectangles
    coords_array = np.vstack((np.array(coords_x), np.array(coords_y))).T

    xs = [random.randint(edge_dist_avoid, img_dims[0] - edge_dist_avoid)
          for x in range(num_data_try)]
    ys = [random.randint(edge_dist_avoid, img_dims[1] - edge_dist_avoid)
          for x in range(num_data_try)]

    coords_sampled = np.vstack(
        (np.array(xs),
         np.array(ys))).T

    indices_to_remove, _, _ = radius_search(
        coords_sampled, coords_array, mindistance)

    coords_sampled = np.delete(coords_sampled, indices_to_remove, axis=0)

    if coords_sampled.shape[0] > num_data:
        keep_rows = random.sample(range(coords_sampled.shape[0]), num_data)
        coords_sampled = coords_sampled[keep_rows, ]

    offset_use = list((x // 2 for x in dimensions))
    x_coords = [x - offset_use[0] for x in coords_sampled[:, 0]]
    y_coords = [x - offset_use[1] for x in coords_sampled[:, 1]]

    chunk_gen = chunk_image_on_position(
        img_arr, x_coords,
        y_coords,
        dimensions=dimensions)

    for _, _, chunk in chunk_gen:
        yield(chunk)
