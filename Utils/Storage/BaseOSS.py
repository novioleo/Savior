import io
from abc import ABC, abstractmethod

import cv2
import msgpack
import msgpack_numpy as m
import numpy as np
from PIL import Image

from Utils.Exceptions import ImageFileSizeAbnormalException, ImageClassNotSupportToEncodeException
from Utils.misc import convert_pil_to_numpy


class CloudObjectStorage(ABC):

    def __init__(self, _endpoint, _access_key, _secret_key):
        self.endpoint = _endpoint
        self.access_key = _access_key
        self.secret_key = _secret_key

    @abstractmethod
    def create_bucket(self, _bucket_name):
        pass

    @abstractmethod
    def download_data(self, _bucket_name, _object_path):
        pass

    @abstractmethod
    def upload_data(self, _bucket_name, _object_path, _to_upload_object_bytes):
        pass

    @abstractmethod
    def get_retrieve_url(self, _bucket_name, _object_path, _expire_seconds=86400 * 7):
        pass

    @abstractmethod
    def check_file_exist(self, _bucket_name, _object_path):
        pass

    @staticmethod
    def _image_object_encode(_m_img, _enable_compress, _quality_rate):
        """
        对图像object进行编码

        :param _m_img:  图像（numpy或者PIL）
        :param _enable_compress:    是否需要压缩
        :param _quality_rate:   具体压缩比例(0-100)
        :return:    字节流
        """
        if isinstance(_m_img, np.ndarray):
            if not _enable_compress:
                to_upload_img_bytes = io.BytesIO(cv2.imencode('.png', _m_img)[1])
            else:
                # webp的效率是png的1/3，但是比类似质量的图像小10倍
                to_upload_img_bytes = io.BytesIO(
                    cv2.imencode('.webp', _m_img, [cv2.IMWRITE_WEBP_QUALITY, _quality_rate])[1]
                )
        elif isinstance(_m_img, Image.Image):
            to_upload_img_bytes = io.BytesIO()
            if not _enable_compress:
                _m_img.save(to_upload_img_bytes, format='PNG')
            else:
                _m_img.save(to_upload_img_bytes, format='WEBP', quality=_quality_rate)
        else:
            raise ImageClassNotSupportToEncodeException(f'{type(_m_img)} not support now')
        return to_upload_img_bytes

    @staticmethod
    def _image_object_decode(_img_object_bytes, _image_size_threshold=10):
        """
        对图像进行解码

        :param _img_object_bytes:     图像字节
        :param _image_size_threshold:   图像大小阈值，单位KB
        :return:    解码后的numpy数组
        """
        image_file_stream = io.BytesIO(_img_object_bytes)
        img_pil = Image.open(image_file_stream)
        request_image = convert_pil_to_numpy(img_pil)
        if _image_size_threshold and request_image.nbytes < 1024 * _image_size_threshold:
            raise ImageFileSizeAbnormalException('图像过小，可能不是正常图片')
        return request_image

    @staticmethod
    def _general_numpy_object_encode(_to_encode_array):
        return io.BytesIO(msgpack.packb(_to_encode_array, default=m.encode))

    @staticmethod
    def _general_numpy_object_decode(_to_decode_bytes):
        return msgpack.unpackb(_to_decode_bytes, object_hook=m.decode)

    def download_image_file(self, _bucket_name, _object_path, _image_size_threshold=None):
        return self._image_object_decode(self.download_data(_bucket_name, _object_path), _image_size_threshold)

    def download_numpy_array(self, _bucket_name, _object_path):
        return self._general_numpy_object_decode(self.download_data(_bucket_name, _object_path))

    def upload_image_file(self, _bucket_name, _object_path, _image, _enable_compress=True, _quality_rate=90):
        return self.upload_data(_bucket_name, _object_path + ('.webp' if _enable_compress else '.png'),
                                self._image_object_encode(_image, _enable_compress, _quality_rate)
                                )

    def upload_numpy_array(self, _bucket_name, _object_path, _np_array):
        return self.upload_data(_bucket_name, _object_path,
                                self._general_numpy_object_encode(_np_array)
                                )
