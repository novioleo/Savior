from abc import ABC

import numpy as np

from Operators.DummyAlgorithmWithModel import DummyAlgorithmWithModel
from Operators.ExampleTextRecognizeOperator.CTCDecoder import CTCLabelConverter
from Utils.GeometryUtils import resize_with_height, center_pad_image_with_specific_base, force_convert_image_to_bgr
from Utils.InferenceHelpers import TritonInferenceHelper


class TextRecognizeOperator(DummyAlgorithmWithModel, ABC):
    """
    可以根据自己的需求定制自己的文本识别模型
    """
    name = "文本识别"
    __version__ = 'v1.0.20210315'

    def __init__(self, _inference_config, _alphabet_config_name, _is_test):
        super().__init__(_inference_config, _is_test)
        self.ctc_decoder = CTCLabelConverter(_alphabet_config_name)


class GeneralCRNN(TextRecognizeOperator):
    """
    面向自然场景的CRNN
    """
    name = "基于CRNN的自然场景文本识别"
    __version__ = 'v1.0.20210315'

    def get_inference_helper(self):
        if self.inference_config['name'] == 'triton':
            support_backbone_type = {'mbv3', 'res34'}
            backbone_type = 'res34'
            if 'backbone_type' in self.inference_config and \
                    self.inference_config['backbone_type'] in support_backbone_type:
                backbone_type = self.inference_config['backbone_type']
            else:
                print(f'crnn use the default backbone:{backbone_type}')
            inference_helper = TritonInferenceHelper(
                f'CRNN_{backbone_type}',
                self.inference_config['triton_url'],
                self.inference_config['triton_port'],
                f'CRNN_{backbone_type}',
                1
            )
            inference_helper.add_image_input('INPUT__0', (32, -1, 3), '识别用的图像',
                                             ([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]))
            inference_helper.add_output('OUTPUT__0', (-1, 1), '识别的max')
            inference_helper.add_output('OUTPUT__1', (-1, 1), '识别的argmax的结果')
            self.inference_helper = inference_helper
        else:
            raise NotImplementedError(
                f"{self.inference_config['name']} helper for crnn text recognize not implement")

    def execute(self, _image):
        to_return_result = {
            'text': '',
            'probability': []
        }
        resized_image = resize_with_height(_image, 32)
        padded_image = center_pad_image_with_specific_base(resized_image, _width_base=4)
        candidate_image = force_convert_image_to_bgr(padded_image)
        if isinstance(self.inference_helper, TritonInferenceHelper):
            result = self.inference_helper.infer(_need_tensor_check=False, INPUT__0=candidate_image.astype(np.float32))
            predict_index, predict_score = result['OUTPUT__1'], result['OUTPUT__0']
        else:
            raise NotImplementedError(f"{self.inference_helper.type_name} helper for crnn not implement")
        decode_result = self.ctc_decoder.decode(predict_index, predict_score)[0]
        to_return_result['text'] = decode_result[0]
        to_return_result['probability'] = decode_result[1]
        return to_return_result


if __name__ == '__main__':
    from argparse import ArgumentParser
    import cv2
    from pprint import pprint

    ag = ArgumentParser('Text Recognize Example')
    ag.add_argument('-i', '--image_path', dest='image_path', type=str, required=True, help='本地图像路径')
    ag.add_argument('-u', '--triton_url', dest='triton_url', type=str, required=True, help='triton url')
    ag.add_argument('-p', '--triton_port', dest='triton_port', type=int, default=8001, help='triton grpc 端口')
    ag.add_argument('-b', '--backbone', dest='backbone', choices=['mbv3', 'res34'], default='res34',
                    help='crnn的backbone')
    args = ag.parse_args()
    img = cv2.imread(args.image_path)
    crnn_handler = GeneralCRNN({
        'name': 'triton',
        'backbone_type': args.backbone,
        'triton_url': args.triton_url,
        'triton_port': args.triton_port}, 'common', True)
    pprint(crnn_handler.execute(img))
