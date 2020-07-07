import numpy as np
from jspp_imageutils.annotations.point_annot_tools import radius_search


def make_test_data(n, i, j, r=100):
    X = np.random.rand(i, n) * r - r / 2
    Y = np.random.rand(j, n) * r - r / 2
    return X, Y


def test_radius_search_out_dims():
    pass


def test_radius_search():
    X, Y = make_test_data(3, 12, 24, r=2)
    print(radius_search(X, Y, radius=2))
    print(radius_search(X+5., Y, radius=2))

    pass
