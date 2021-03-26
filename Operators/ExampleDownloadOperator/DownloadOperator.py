import os
import traceback as tb
from collections import OrderedDict

from skimage import io
import cv2

from Operators.DummyOperator import DummyOperator
from Utils.Exceptions import ImageDownloadTimeoutException, ImageFileSizeAbnormalException, CustomException, \
    ImageFormatNotSupportException, ConsumerAlgorithmUncatchException
from Utils.Storage import DummyOSS
from Utils.misc import get_uuid_name, get_date_string


class ImageDownloadOperator(DummyOperator):
    """
    将图像从网上下载下来，然后存储在OSS中，方便集群中的worker进行取用。
    """

    name = "图像下载"
    __version__ = 'v1.0.20210312'

    def __init__(self, _is_test):
        super().__init__(_is_test)

    def execute(
            self, _to_download_url, _oss_helper,
            _timeout=30, _image_size_threshold=10,

    ):
        to_return_result = OrderedDict()
        try:
            request_image = cv2.cvtColor(io.imread(_to_download_url), cv2.COLOR_RGB2BGR)
            if request_image is None:
                raise ImageFormatNotSupportException(
                    f'image:{_to_download_url} format not support,cannot decode by opencv')
            if len(request_image.shape) == 3:
                image_h, image_w, image_c = request_image.shape
            else:
                image_h, image_w = request_image.shape[:2]
                image_c = 1
            if _image_size_threshold is not None and request_image.nbytes < _image_size_threshold * 1024:
                raise ImageFileSizeAbnormalException(
                    f'image:{_to_download_url} is small than threshold,it may not be a normal picture')
            file_name = get_uuid_name()
            oss_path = os.path.join(get_date_string(), file_name)
            # 存储原始图像
            saved_path = _oss_helper.upload_image_file('downloaded-image', oss_path, request_image,
                                                       _enable_compress=False)
            to_return_result['bucket_name'] = 'downloaded-image'
            to_return_result['saved_path'] = saved_path
            to_return_result['image_height'] = image_h
            to_return_result['image_width'] = image_w
            to_return_result['image_channel'] = image_c
            return to_return_result
        except TimeoutError as te:
            raise ImageDownloadTimeoutException(f'{_to_download_url} download timeout')
        except CustomException as ce:
            raise ce
        except Exception as e:
            raise ConsumerAlgorithmUncatchException(tb.format_exc())


if __name__ == '__main__':
    from pprint import pprint
    from argparse import ArgumentParser

    ag = ArgumentParser('Image Download Example')
    ag.add_argument('-i', '--image', type=str, dest='image',
                    default='https://www.baidu.com/img/flexible/logo/pc/result@2.png',
                    help='待下载图片')
    args = ag.parse_args()
    oss_helper = DummyOSS(None, None, None)
    image_url = args.image
    image_download_operator = ImageDownloadOperator(True)
    download_result = image_download_operator.execute(image_url, oss_helper)
    pprint(download_result)
