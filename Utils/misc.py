import datetime
import uuid
import numpy as np
import cv2
from PIL import Image

from Utils.Exceptions import ImageFormatNotSupportException


def get_date_string():
    return datetime.date.today().strftime('%Y%m%d')


def get_uuid_name():
    return str(uuid.uuid4()).replace('-', '')


def convert_pil_to_numpy(_pil_image: Image.Image):
    """
    将pil的图像转换到numpy，并且保证通道的正确

    Args:
        _pil_image: 待转换图像

    Returns:    图像的numpy数组

    """
    img_np = np.asarray(_pil_image)
    len_bands = len(_pil_image.getbands())
    if len_bands == 3:
        to_return_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif len_bands == 4:
        to_return_image = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGRA)
    elif len_bands == 1:
        to_return_image = img_np.copy()
    else:
        raise ImageFormatNotSupportException(f'image mode [{"".join(_pil_image.getbands())}] not support now')
    return to_return_image
