from nptyping import NDArray, UInt
from typing import Union, Any

ImgArray = NDArray[(Any, Any, Any), float]
MaskArray = NDArray[(Any, Any), UInt]

ImgBatch = NDArray[(Any, Any, Any, Any), float]
MaskBatch = NDArray[(Any, Any, Any, 1), UInt]

GenImgBatch = Union[ImgBatch, MaskBatch]
GenImgArray = Union[ImgArray, MaskArray]
