import base64
import os
import re
import traceback as tb
from io import BytesIO

from PIL import Image

from Operators.DummyOperator import DummyOperator
from Utils.Exceptions import CustomException, ConsumerAlgorithmUncatchException
from Utils.Storage import CloudObjectStorage
from Utils.misc import get_uuid_name, get_date_string, convert_pil_to_numpy


class ImageParseFromBase64(DummyOperator):

    def __init__(self, _is_test):
        super().__init__(_is_test)
        self.bucket_name = 'base64_images'

    def execute(self, _base64_string, _oss_handler: CloudObjectStorage = None):
        try:
            base64_data = re.sub('^data:image/.+;base64,', '', _base64_string)
            byte_data = base64.b64decode(base64_data)
            image_data = BytesIO(byte_data)
            decoded_image = Image.open(image_data)
            image_c = len(decoded_image.getbands())
            image_h = decoded_image.height
            image_w = decoded_image.width
            decoded_image_np = convert_pil_to_numpy(decoded_image)
            saved_path = ''
            target_bucket_name = ''
            if _oss_handler:
                file_name = get_uuid_name()
                oss_path = os.path.join(get_date_string(), file_name)
                saved_path = _oss_handler.upload_image_file(self.bucket_name, oss_path, decoded_image,
                                                            _enable_compress=False)
                target_bucket_name = self.bucket_name
            to_return_result = dict()
            to_return_result['bucket_name'] = target_bucket_name
            to_return_result['saved_path'] = saved_path
            to_return_result['image_height'] = image_h
            to_return_result['image_width'] = image_w
            to_return_result['image_channel'] = image_c
            to_return_result['image'] = decoded_image_np
            return to_return_result
        except CustomException as ce:
            raise ce
        except Exception as e:
            raise ConsumerAlgorithmUncatchException(tb.format_exc())


if __name__ == '__main__':
    import cv2
    import argparse

    ag = argparse.ArgumentParser('Base64 String Decode To Image Test')
    ag.add_argument('--file', type=str, required=True, help='base64字符串所在文本文件路径')
    args = ag.parse_args()
    base64_string_file = args.file
    image_decode_op = ImageParseFromBase64(True)
    with open(base64_string_file, mode='r', encoding='utf-8') as to_read:
        base64_string = to_read.read()
    decode_result = image_decode_op.execute(base64_string, None)
    cv2.imshow('decode_image', decode_result['image'])
    cv2.waitKey(0)
