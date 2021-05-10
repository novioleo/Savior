import os
import traceback as tb
from abc import ABC
from collections import OrderedDict
from io import BytesIO

import requests
from PIL import Image

from Operators.DummyOperator import DummyOperator
from Utils.Exceptions import ImageDownloadTimeoutException, ImageFileSizeAbnormalException, CustomException, \
    ImageFormatNotSupportException, ConsumerAlgorithmUncatchException, DownloadURLTimeoutException, \
    DownloadURLNotAvailableException
from Utils.Storage import CloudObjectStorage
from Utils.misc import get_uuid_name, get_date_string, convert_pil_to_numpy


class DownloadOperator(DummyOperator, ABC):
    name = '下载'
    __version__ = 'v1.0.20210406'

    def __init__(self, _is_test, _timeout=30, _bucket_name='download'):
        super().__init__(_is_test)
        self.timeout = _timeout
        self.bucket_name = _bucket_name

    def download_url(self, _to_download_url, _chunk_size=1024 * 1):
        to_return_bytes = BytesIO()
        try:
            download_response = requests.get(_to_download_url, timeout=self.timeout)
            if download_response.status_code != 200:
                raise DownloadURLNotAvailableException('download fail')
            for m_chuck in download_response.iter_content(chunk_size=_chunk_size):
                to_return_bytes.write(m_chuck)
        except requests.exceptions.Timeout as te:
            raise DownloadURLTimeoutException
        except requests.exceptions.InvalidURL as ie:
            raise DownloadURLNotAvailableException
        return to_return_bytes


class ImageDownloadOperator(DownloadOperator):
    """
    将图像从网上下载下来，可以存储在OSS中，方便集群中的worker进行取用。
    """

    name = "图像下载"
    __version__ = 'v1.0.20210406'

    def __init__(self, _is_test, _timeout=30, _bucket_name='downloaded-image'):
        super().__init__(_is_test, _timeout, _bucket_name)

    def execute(
            self, _to_download_url, _oss_helper: CloudObjectStorage = None,
            _image_size_threshold=10,

    ):
        """
        下载指定url的图像文件

        Args:
            _to_download_url:   待下载的图像url
            _oss_helper:    oss helper用于存储下载好的数据，可以为空
            _image_size_threshold:  图像字节数(KB)限制，如果低于阈值会异常

        Returns:    下载完成的结果

        """
        to_return_result = OrderedDict()
        try:
            if _to_download_url.startswith('http'):
                download_result_io = self.download_url(_to_download_url, 1024 * 2)
                request_image = Image.open(download_result_io)
            else:
                if os.path.exists(_to_download_url):
                    request_image = Image.open(_to_download_url)
                else:
                    request_image = None
            if request_image is None:
                raise ImageFormatNotSupportException(
                    f'image:{_to_download_url} format not support,cannot decode by PILLOW')
            image_c = len(request_image.getbands())
            image_h = request_image.height
            image_w = request_image.width
            request_image_np = convert_pil_to_numpy(request_image)
            if _image_size_threshold is not None and request_image_np.nbytes < _image_size_threshold * 1024:
                raise ImageFileSizeAbnormalException(
                    f'image:{_to_download_url} is small than threshold,it may not be a normal picture')
            # 有些情况是不需要存储的，可能直接就用了。
            if _oss_helper:
                file_name = get_uuid_name()
                oss_path = os.path.join(get_date_string(), file_name)
                # 存储原始图像
                saved_path = _oss_helper.upload_image_file(self.bucket_name, oss_path, request_image,
                                                           _enable_compress=False)
            else:
                saved_path = ''
            to_return_result['bucket_name'] = self.bucket_name
            to_return_result['saved_path'] = saved_path
            to_return_result['image_height'] = image_h
            to_return_result['image_width'] = image_w
            to_return_result['image_channel'] = image_c
            to_return_result['image'] = request_image_np
            return to_return_result
        except requests.exceptions.ConnectionError as connect_error:
            raise DownloadURLNotAvailableException(f'{_to_download_url} cannot reach')
        except TimeoutError as te:
            raise ImageDownloadTimeoutException(f'{_to_download_url} download timeout')
        except CustomException as ce:
            raise ce
        except Exception as e:
            raise ConsumerAlgorithmUncatchException(tb.format_exc())


if __name__ == '__main__':
    from pprint import pprint
    from argparse import ArgumentParser
    from Utils.Storage import DummyOSS, get_oss_handler

    ag = ArgumentParser('Image Download Example')
    ag.add_argument('-i', '--image', type=str, dest='image',
                    default='https://www.baidu.com/img/flexible/logo/pc/result@2.png',
                    help='待下载图片')
    args = ag.parse_args()
    oss_helper = DummyOSS(None, None, None)
    image_url = args.image
    image_download_operator = ImageDownloadOperator(True)
    download_result = image_download_operator.execute(image_url, oss_helper)
    download_img = oss_helper.download_image_file(download_result['bucket_name'], download_result['saved_path'])
    print(download_img.shape[:2])
    pprint(download_result)
