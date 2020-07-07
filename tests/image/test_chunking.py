import numpy as np
import PIL

from jspp_imageutils.image.chunking import chunk_image_generator
from jspp_imageutils.image.stitching import stitch_images


def test_chunking_works_on_arrays():
    fakeimg = np.array(range(100))
    fakeimg = fakeimg.reshape(10, 10)

    gen = chunk_image_generator(fakeimg, (2, 2), (2, 2))
    chunk = next(gen)

    img_chunk = chunk[2]
    # assert isinstance(img_chunk, NDArray[(2, 2), Float[32]])
    assert img_chunk.shape == (2, 2)
    # assert isinstance(img_chunk, ImgArray)


def test_chunking_works_on_cv2_imported_images(shared_datadir):

    filepath = shared_datadir / "ihc_1.png"

    img = PIL.Image.open(filepath)

    img_gen = chunk_image_generator(img, (10, 10), (10, 10), 32)
    chunk = next(img_gen)
    img_chunk = chunk[2]
    img_chunk_shape = img_chunk.shape

    # assert isinstance(img_chunk, NDArray[(10, 10, 3), UInt])
    assert img_chunk_shape == (10, 10, 3)
    assert isinstance(img_chunk, np.ndarray)


def test_stitching_and_chunking_equality_on_arrays():
    fakeimg = np.array(range(100))
    fakeimg = fakeimg.reshape(10, 10)

    gen = chunk_image_generator(fakeimg, (2, 2), (2, 2))
    img2 = stitch_images(gen, (10, 10), (2, 2))

    comparisson = fakeimg == img2
    assert comparisson.all()

    gen = chunk_image_generator(
        fakeimg, (2, 2), (1, 1),
        warn_leftovers=False)
    img2 = stitch_images(gen, (10, 10), (1, 1))

    comparisson = fakeimg == img2
    assert comparisson.all()

    fakeimg = np.array(range(10000))
    fakeimg = fakeimg.reshape(100, 100)
    gen = chunk_image_generator(
        fakeimg, (10, 10), (1, 1),
        warn_leftovers=False)
    reconstructed = stitch_images(gen, (100, 100), (1, 1))
    comparisson = fakeimg == reconstructed
    assert comparisson.all()


def test_stitching_and_chunking_operations():
    fakeimg = np.array(range(100))
    fakeimg = fakeimg.reshape(10, 10)
    gen = chunk_image_generator(fakeimg, (2, 2), (2, 2))
    img2 = stitch_images(gen, (10, 10), (2, 2), operation="add")

    # In cases where the dispalcement is the same length of chunks
    # it shoudld be the same
    comparisson = fakeimg == img2
    assert comparisson.all()

    fakeimg = np.ones(100)
    fakeimg = fakeimg.reshape(10, 10)
    gen = chunk_image_generator(
        fakeimg, (2, 2), (1, 1),
        warn_leftovers=False)
    img2 = stitch_images(
        gen, (10, 10),
        (1, 1), operation="add")

    inner_comparisson = img2[2:-2, 2:-2] == 4
    assert inner_comparisson.all()

    outer_comparisson = (img2 <= 4) & (img2 > 0)
    assert outer_comparisson.all()


def test_stitching_works_on_3d_arrays():
    fakeimg = np.array(range(300))
    fakeimg = fakeimg.reshape(10, 10, 3)
    gen = chunk_image_generator(fakeimg, (2, 2), (2, 2))
    img2 = stitch_images(gen, (10, 10, 3), (2, 2), operation="replace")

    # In cases where the dispalcement is the same length of chunks
    # it shoudld be the same
    comparisson = fakeimg == img2
    assert comparisson.all()
