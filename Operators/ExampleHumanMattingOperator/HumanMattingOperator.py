from abc import ABC
import numpy as np
from Operators.DummyAlgorithmWithModel import DummyAlgorithmWithModel
from Utils.GeometryUtils import force_convert_image_to_bgr
from Utils.InferenceHelpers import TritonInferenceHelper
import cv2


class HumanMattingOperator(DummyAlgorithmWithModel, ABC):
    __version__ = 'v1.0.20210413'
    name = '人体抠图'


class HumanMattingWithUNet(HumanMattingOperator):

    """
    精度高，但是由于是直接进行的encoder decoder，在decoder阶段的跨层连接很多且很大，所以显存占用很多
    """

    name = '基于u^2net的人体抠图'
    __version__ = 'v1.0.20210413'

    def __init__(self, _inference_config, _is_test):
        super().__init__(_inference_config, _is_test)
        self.target_shape = (1280, 720)

    def get_inference_helper(self):
        if self.inference_config['name'] == 'triton':
            inference_helper = TritonInferenceHelper('HumanMattingUnet',
                                                     self.inference_config['triton_url'],
                                                     self.inference_config['triton_port'],
                                                     'HumanMattingUnet', 1)
            inference_helper.add_image_input('INPUT__0', (1280, 720, 3), '分割用rgb图像',
                                             ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375]))
            inference_helper.add_output('OUTPUT__0', (1, 720, 1280), '分类结果')
            self.inference_helper = inference_helper
        else:
            raise NotImplementedError(
                f"{self.inference_config['name']} helper for unet human matting not implement")

    @staticmethod
    def normalize_matting_result(_matting_result):
        max_value = _matting_result.max()
        min_value = _matting_result.min()
        if max_value == min_value:
            return _matting_result
        to_return_normalized_result = (_matting_result - min_value) / (max_value - min_value)
        return to_return_normalized_result

    def execute(self, _image):
        to_return_result = {
            'matting_alpha': np.zeros((_image.shape[1], _image.shape[0]), dtype=np.float32),
        }
        original_h, original_w = _image.shape[:2]
        resized_image = cv2.resize(_image, self.target_shape)
        candidate_image = cv2.cvtColor(force_convert_image_to_bgr(resized_image), cv2.COLOR_BGR2RGB)
        if isinstance(self.inference_helper, TritonInferenceHelper):
            result = self.inference_helper.infer(_need_tensor_check=False, INPUT__0=candidate_image.astype(np.float32))
            matting_result = result['OUTPUT__0'].squeeze()
        else:
            raise NotImplementedError(
                f"{self.inference_helper.type_name} helper for unet human matting not implement")
        normalize_matting_result = self.normalize_matting_result(matting_result)
        # 恢复resize
        resize_back_matting_result = cv2.resize(normalize_matting_result, (original_w, original_h),
                                                interpolation=cv2.INTER_LINEAR)
        to_return_result['matting_alpha'] = resize_back_matting_result
        return to_return_result


if __name__ == '__main__':
    from argparse import ArgumentParser

    ag = ArgumentParser('Human Matting Example')
    ag.add_argument('-i', '--image_path', dest='image_path', type=str, required=True, help='本地图像路径')
    ag.add_argument('-u', '--triton_url', dest='triton_url', type=str, required=True, help='triton url')
    ag.add_argument('-p', '--triton_port', dest='triton_port', type=int, default=8001, help='triton grpc 端口')
    args = ag.parse_args()
    # 假设图中必然有人
    img = cv2.imread(args.image_path)
    matting_operator = HumanMattingWithUNet({
        'name': 'triton',
        'triton_url': args.triton_url,
        'triton_port': args.triton_port
    }, True)
    matting_result = matting_operator.execute(img)
    alpha = np.uint8(matting_result['matting_alpha'] * 255)
    b, g, r = cv2.split(img)
    image_with_alpha = cv2.merge([b, g, r, alpha])
    cv2.imwrite('image_with_alpha.png', image_with_alpha)
