from abc import ABC
from enum import Enum
import cv2
import numpy as np

from Operators.DummyAlgorithmWithModel import DummyAlgorithmWithModel
from Utils.GeometryUtils import force_convert_image_to_bgr
from Utils.InferenceHelpers import TritonInferenceHelper


class TextImageOrientation(Enum):
    """
    文本朝向四个枚举类
    """
    ORIENTATION_0: int = 0
    ORIENTATION_90: int = 1
    ORIENTATION_180: int = 2
    ORIENTATION_270: int = 3
    ORIENTATION_UNRELIABLE: int = 4


class TextOrientationOperator(DummyAlgorithmWithModel, ABC):
    name = 'TextOrientationDetect'
    __version__ = 'v1.0.20210411'

    def __init__(self, _inference_config, _is_test):
        super().__init__(_inference_config, _is_test)


class GeneralTextOrientationOperator(TextOrientationOperator):
    """
    分类准确率不能保证
    """

    name = '常规文本图像的方向（正向和倒向）'
    __version__ = 'v1.0.20210411'

    def __init__(self, _inference_helper, _is_test, _threshold=0.8):
        super().__init__(_inference_helper, _is_test)
        self.target_shape = (192, 48)
        # confidence threshold
        self.threshold = _threshold

    def get_inference_helper(self):
        if self.inference_config['name'] == 'triton':
            inference_helper = TritonInferenceHelper(
                'TextOrientationClassification',
                self.inference_config['triton_url'],
                self.inference_config['triton_port'],
                'TextOrientationClassification',
                1
            )
            inference_helper.add_image_input('INPUT__0', (192, 48, 3), '识别用的图像',
                                             ([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]))
            inference_helper.add_output('OUTPUT__0', (2,), '方向的分类')
            self.inference_helper = inference_helper
        else:
            raise NotImplementedError(
                f"{self.inference_config['name']} helper for text orientation not implement")

    def execute(self, _image):
        to_return_result = {
            'orientation': TextImageOrientation.ORIENTATION_0,
        }
        resized_image = cv2.resize(_image, self.target_shape)
        candidate_image = force_convert_image_to_bgr(resized_image)
        cv2.imwrite('candidate_image.png', candidate_image)
        if isinstance(self.inference_helper, TritonInferenceHelper):
            result = self.inference_helper.infer(_need_tensor_check=False, INPUT__0=candidate_image.astype(np.float32))
            classification = result['OUTPUT__0'].squeeze()
        else:
            raise NotImplementedError(f"{self.inference_helper.type_name} helper for db not implement")
        target_index = np.argmax(classification)
        if classification[target_index] > self.threshold:
            if target_index == 0:
                to_return_result['orientation'] = TextImageOrientation.ORIENTATION_0
            else:
                to_return_result['orientation'] = TextImageOrientation.ORIENTATION_180
        else:
            to_return_result['orientation'] = TextImageOrientation.ORIENTATION_UNRELIABLE
        return to_return_result


if __name__ == '__main__':
    from argparse import ArgumentParser

    ag = ArgumentParser('Text Orientation Classification Example')
    ag.add_argument('-i', '--image_path', dest='image_path', type=str, required=True, help='本地图像路径')
    ag.add_argument('-u', '--triton_url', dest='triton_url', type=str, required=True, help='triton url')
    ag.add_argument('-p', '--triton_port', dest='triton_port', type=int, default=8001, help='triton grpc 端口')
    args = ag.parse_args()
    img = cv2.imread(args.image_path)
    orientation_classification_handler = GeneralTextOrientationOperator({
        'name': 'triton',
        'triton_url': args.triton_url,
        'triton_port': args.triton_port}, True)
    print(orientation_classification_handler.execute(img))
