import io
from abc import ABC, abstractmethod

from minio import Minio
import numpy as np
import cv2
import msgpack
import msgpack_numpy as m
import os

from Deployment.server_config import OSS_TYPE, OSS_INFO
from Utils.Exceptions import ImageFileSizeAbnormalException, ObjectNotFoundOnOSS


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

    @staticmethod
    def _image_object_encode(_m_img, _enable_compress, _quality_rate):
        """
        对图像object进行编码

        :param _m_img:  图像numpy数组
        :param _enable_compress:    是否需要压缩
        :param _quality_rate:   具体压缩比例
        :return:    字节流
        """
        if not _enable_compress:
            to_upload_img_bytes = io.BytesIO(cv2.imencode('.png', _m_img)[1])
        else:
            # webp的效率是png的1/3，但是比类似质量的图像小10倍
            to_upload_img_bytes = io.BytesIO(
                cv2.imencode('.webp', _m_img, [cv2.IMWRITE_WEBP_QUALITY, _quality_rate])[1]
            )
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
        m_image_file_buffer = image_file_stream.read()
        request_image = cv2.imdecode(np.frombuffer(m_image_file_buffer, np.uint8), -1)
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


class DummyOSS(CloudObjectStorage):
    """

    单机测试用oss，全程存储在本地

    """

    def __init__(self, _endpoint, _access_key, _secret_key):
        super().__init__(_endpoint, _access_key, _secret_key)
        self.temp_directory_path = '/tmp/DummyOSS-temp-directory'
        os.makedirs(self.temp_directory_path, exist_ok=True)

    def create_bucket(self, _bucket_name):
        os.makedirs(os.path.join(self.temp_directory_path, _bucket_name), exist_ok=True)

    def download_data(self, _bucket_name, _object_path):
        target_file_path = os.path.join(self.temp_directory_path, _bucket_name, _object_path)
        if not os.path.exists(target_file_path):
            raise ObjectNotFoundOnOSS(target_file_path + ' not found')
        with open(target_file_path, mode='rb') as to_read:
            return to_read.read()

    def upload_data(self, _bucket_name, _object_path, _to_upload_object_bytes):
        target_file_path = os.path.join(self.temp_directory_path, _bucket_name, _object_path)
        os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
        with open(target_file_path, mode='wb') as to_write:
            to_write.write(_to_upload_object_bytes.read())
        return _object_path


class MinioOSS(CloudObjectStorage):

    def __init__(self, _endpoint, _access_key, _secret_key, _region=None):
        super().__init__(_endpoint, _access_key, _secret_key)
        self.region = _region
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            region=self.region,
            secure=False,
        )

    def create_bucket(self, _bucket_name):
        if not self.client.bucket_exists(_bucket_name):
            self.client.make_bucket(_bucket_name, location=self.region)

    def download_data(self, _bucket_name, _object_path):
        try:
            response = self.client.get_object(_bucket_name, _object_path)
            data_bytes = response.data
            return data_bytes
        except Exception as e:
            raise ObjectNotFoundOnOSS(os.path.join(_bucket_name, _object_path) + ' not found')
        finally:
            response.close()
            response.release_conn()

    def upload_data(self, _bucket_name, _object_path, _to_upload_object_bytes):
        return self.client.put_object(_bucket_name, _object_path, _to_upload_object_bytes, -1,
                                      part_size=5 * 1024 * 1024).object_name


def get_oss_handler():
    if OSS_TYPE == 'MINIO':
        to_return_handler = MinioOSS(
            _endpoint=OSS_INFO['ENDPOINT'],
            _access_key=OSS_INFO['ACCESS_KEY'],
            _secret_key=OSS_INFO['SECRET_KEY'],
            _region=OSS_INFO['REGION'],
        )
    elif OSS_TYPE == 'DUMMY':
        to_return_handler = DummyOSS(
            None, None, None
        )
    else:
        raise NotImplementedError(f'oss client {OSS_TYPE} not implement')
    return to_return_handler


if __name__ == '__main__':
    import os
    from argparse import ArgumentParser

    ag = ArgumentParser('OSS Test Example')
    ag.add_argument('--end_point_url', required=True, type=str, help='OSS服务器的URL')
    ag.add_argument('--end_point_port', required=True, type=str, help='OSS服务器的port')
    ag.add_argument('--backend_type', required=True, choices=['minio', 'dummy'], type=str, help='OSS服务器的类型')
    ag.add_argument('-a', '--access_key', type=str, required=True, help='access key')
    ag.add_argument('-s', '--secret_key', type=str, required=True, help='secret key')
    ag.add_argument('-r', '--region', type=str, default=None, help='地域')
    ag.add_argument('-b', '--bucket_name', type=str, default='test_bucket', help='bucket名称')
    ag.add_argument('-i', '--image_path', type=str, required=True, help='测试用的图像路径')
    args = ag.parse_args()
    if args.backend_type == 'minio':
        oss_operator = MinioOSS(
            f'{args.end_point_url}:{args.end_point_port}',
            args.access_key,
            args.secret_key,
            args.region
        )
    elif args.backend_type == 'dummy':
        oss_operator = DummyOSS(
            None, None, None
        )
    else:
        raise NotImplementedError(f'{args.backend_type} not implement')
    bucket_name = args.bucket_name
    oss_operator.create_bucket(bucket_name)
    img = cv2.imread(args.image_path)
    resize_img = cv2.resize(img, (128, 128))
    upload_image_path = os.path.join('test_images', 'image')
    upload_image_path = oss_operator.upload_image_file(
        bucket_name, upload_image_path, resize_img, True, 90
    )
    image_numpy = oss_operator.download_image_file(bucket_name, upload_image_path)
    cv2.imwrite('resize_image.png', image_numpy)
    test_numpy_array = [
        np.random.randint(3, 5, (3, 3)).astype(np.uint8),
        np.random.uniform(0, 1, (5, 5))
    ]
    upload_test_numpy_path = os.path.join('test_numpy_arrays', 'test_array')
    oss_operator.upload_numpy_array(bucket_name, upload_test_numpy_path, test_numpy_array)
    download_test_numpy_array = oss_operator.download_numpy_array(bucket_name, upload_test_numpy_path)
    for m_array, m_download_array in zip(test_numpy_array, download_test_numpy_array):
        assert np.sum(m_array - m_download_array) == 0, 'there is different'
