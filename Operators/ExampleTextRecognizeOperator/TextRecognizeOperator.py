from abc import ABC

import numpy as np

from Operators.DummyAlgorithm import DummyAlgorithm
from Operators.ExampleTextRecognizeOperator.CTCDecoder import CTCLabelConverter
from Utils.GeometryUtils import resize_with_height, center_pad_image_with_specific_base
from Utils.InferenceHelpers.BaseInferenceHelper import TritonInferenceHelper


class TextRecognizeOperator(DummyAlgorithm, ABC):
    """
    可以根据自己的需求定制自己的文本识别模型
    """
    name = "文本识别"
    __version__ = 'v1.0.20210315'

    def __init__(self, _inference_helper, _alphabet_config_name, _is_test):
        super().__init__(_is_test)
        self.inference_helper = _inference_helper
        self.ctc_decoder = CTCLabelConverter(_alphabet_config_name)


class GeneralCRNN(TextRecognizeOperator):
    """
    面向自然场景的CRNN
    """
    name = "基于CRNN的自然场景文本识别"
    __version__ = 'v1.0.20210315'

    def __init__(self, _inference_helper, _alphabet_config_name, _is_test):
        super().__init__(_inference_helper, _alphabet_config_name, _is_test)

    def execute(self, _image):
        if isinstance(self.inference_helper, TritonInferenceHelper):
            resized_image = resize_with_height(_image, 32)
            padded_image = center_pad_image_with_specific_base(resized_image, _width_base=4).astype(np.float32)
            result = self.inference_helper.infer(_need_tensor_check=False, INPUT__0=padded_image)
            return self.ctc_decoder.decode(result['OUTPUT__1'], result['OUTPUT__0'])
        else:
            raise NotImplementedError(f"{self.inference_helper.type_name} helper for ncnn not implement")


if __name__ == '__main__':
    import cv2
    from argparse import ArgumentParser
    ag = ArgumentParser('Text Recognize Example')
    ag.add_argument('-i', '--image_path', dest='image_path', type=str, required=True, help='本地图像路径')
    ag.add_argument('-u', '--triton_url', dest='triton_url', type=str, required=True, help='triton url')
    ag.add_argument('-p', '--triton_url', dest='triton_port', type=int, default=8001, help='triton grpc 端口')
    args = ag.parse_args()
    img = cv2.imread(args.image_path)
    crnn_mbv3 = TritonInferenceHelper('crnn_mbv3', args.triton_url, args.triton_port, 'CRNN_mbv3', 1)
    crnn_mbv3.add_image_input('INPUT__0', (32, -1, 3), '识别用的图像', ([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]))
    crnn_mbv3.add_output('OUTPUT__0', (-1, 1), '识别的max')
    crnn_mbv3.add_output('OUTPUT__1', (-1, 1), '识别的argmax的结果')
    crnn_res34 = TritonInferenceHelper('crnn_res34', 'localhost', 8001, 'CRNN_res34', 1)
    crnn_res34.add_image_input('INPUT__0', (32, -1, 3), '识别用的图像', ([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]))
    crnn_res34.add_output('OUTPUT__0', (-1, 1), '识别的max')
    crnn_res34.add_output('OUTPUT__1', (-1, 1), '识别的argmax的结果')
    crnn_mbv3_handler = GeneralCRNN(crnn_mbv3, 'common', True)
    crnn_res34_handler = GeneralCRNN(crnn_res34, 'common', True)
    print(crnn_mbv3_handler.execute(img))
    print(crnn_res34_handler.execute(img))
