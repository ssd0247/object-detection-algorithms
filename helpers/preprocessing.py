from typing import Tuple, Union, Type, Generator, Any

import numpy as np
import cv2
from PIL.JpegImagePlugin import JpegImageFile

# Type[np.ndarray] = cv2 support
# Type[JpegImageFile] = PIL support
Image_Type = Union[Type[np.ndarray], Type[JpegImageFile]]
Image_Type_Any = Any # NOTE: to resolve the mypy error

sliding_window_return_type = Tuple[int, int, Image_Type]


def sliding_window(
    image: Type[np.ndarray],
    step: int,
    ws: Tuple[int, int]
) -> Generator:
    """"""
    
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            yield (x, y, image[y : y+ws[1], x : x+ws[0]])


def _resize(
    image: Image_Type_Any,
    width: Union[int, None] = None,
    height: Union[int, None] = None,
    inter: int = cv2.INTER_AREA,
) -> int:
    """"""
    
    dim = None
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def image_pyramid(
    image: Image_Type_Any,
    scale: float = 1.5,
    minSize: Tuple[int, int] = (224, 224)
) -> Generator:
    """"""

    yield image

    while True:
        w = int(image.shape[1] / scale)
        image = _resize(image, width=w)

        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        yield image