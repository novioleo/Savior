from abc import ABC
import numpy as np
from Operators.DummyAlgorithmWithModel import DummyAlgorithmWithModel
from Utils.GeometryUtils import force_convert_image_to_bgr, resize_with_long_side, center_pad_image_with_specific_base, \
    remove_image_pad
from Utils.InferenceHelpers import TritonInferenceHelper
import cv2


class HumanMattingOperator(DummyAlgorithmWithModel, ABC):
    __version__ = 'v1.0.20210413'
    name = '人体抠图'


class HumanMattingWithUNet(HumanMattingOperator):
    """
    精度高且带有SOD的模式，但是由于是直接进行的encoder decoder，在decoder阶段的跨层连接很多且很大，所以显存占用很多
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
            inference_helper.add_output('OUTPUT__0', (1, 720, 1280), '分割结果')
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


class HumanMattingWithBiSeNet(HumanMattingOperator):
    """
    精度较高且主要以人为准，但是对图像进行进行了缩放，所以边缘会有些差别
    """

    name = '基于bisenetv1+encoder+decoder的人体抠图'
    __version__ = 'v1.0.20210416'

    def __init__(self, _inference_config, _is_test):
        super().__init__(_inference_config, _is_test)
        self.target_shape = (512, 512)

    def get_inference_helper(self):
        if self.inference_config['name'] == 'triton':
            inference_helper = TritonInferenceHelper('HumanMattingBiSe',
                                                     self.inference_config['triton_url'],
                                                     self.inference_config['triton_port'],
                                                     'HumanMattingBiSe', 1)
            inference_helper.add_image_input('INPUT__0', (512, 512, 3), '分割用bgr图像',
                                             ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375]))
            inference_helper.add_output('OUTPUT__0', (4, 512, 512), '分割结果，最后一个通道为alpha信息')
            self.inference_helper = inference_helper
        else:
            raise NotImplementedError(
                f"{self.inference_config['name']} helper for bise human matting not implement")

    def execute(self, _image):
        to_return_result = {
            'matting_alpha': np.zeros((_image.shape[1], _image.shape[0]), dtype=np.float32),
        }
        original_h, original_w = _image.shape[:2]
        resized_image = resize_with_long_side(_image, 512)
        padded_image, (left_margin_ration, top_margin_ratio) = \
            center_pad_image_with_specific_base(resized_image, 512, 512, 0, True)
        candidate_image = force_convert_image_to_bgr(padded_image)
        if isinstance(self.inference_helper, TritonInferenceHelper):
            result = self.inference_helper.infer(_need_tensor_check=False, INPUT__0=candidate_image.astype(np.float32))
            matting_result = result['OUTPUT__0'].squeeze()
        else:
            raise NotImplementedError(
                f"{self.inference_helper.type_name} helper for bise human matting not implement")
        alpha_result = matting_result[3, ...]
        matting_result_without_pad = remove_image_pad(alpha_result, resized_image, left_margin_ration, top_margin_ratio)
        resize_back_matting_result = cv2.resize(matting_result_without_pad, (original_w, original_h),
                                                interpolation=cv2.INTER_LINEAR)
        to_return_result['matting_alpha'] = resize_back_matting_result
        return to_return_result


if __name__ == '__main__':
    from argparse import ArgumentParser

    ag = ArgumentParser('Human Matting Example')
    ag.add_argument('-i', '--image_path', dest='image_path', type=str, required=True, help='本地图像路径')
    ag.add_argument('-u', '--triton_url', dest='triton_url', type=str, required=True, help='triton url')
    ag.add_argument('-p', '--triton_port', dest='triton_port', type=int, default=8001, help='triton grpc 端口')
    ag.add_argument('-t', '--image_type', dest='image_type', choices=['rgba', 'rgb'], default='rgba', help='保存抠图结果的类型rgb和rgba')
    ag.add_argument('-m', '--model_type', dest='model_type', choices=['unet', 'bise'], default='bise',
                    help='matting使用的模型')
    args = ag.parse_args()
    # 假设图中必然有人
    img = cv2.imread(args.image_path)
    if args.model_type == 'unet':
        matting_operator = HumanMattingWithUNet({
            'name': 'triton',
            'triton_url': args.triton_url,
            'triton_port': args.triton_port
        }, True)
    elif args.model_type == 'bise':
        matting_operator = HumanMattingWithBiSeNet({
            'name': 'triton',
            'triton_url': args.triton_url,
            'triton_port': args.triton_port
        }, True)
    else:
        raise Exception(f'{args.model_type} not found')
    matting_result = matting_operator.execute(img)
    alpha = np.uint8(matting_result['matting_alpha'] * 255)
    #抠图结果是rgba形式的
    if args.image_type=='rgba':
        b, g, r = cv2.split(img)
        image_with_alpha = cv2.merge([b, g, r, alpha])
        cv2.imwrite(f'alpha_image_with_{args.model_type}.png', image_with_alpha)
    #抠图结果是rgb形式的
    elif args.image_type=='rgb':
        blurred_mask = cv2.GaussianBlur(alpha, (13, 13), 11)
        blurred_mask = np.repeat(blurred_mask[..., None], 3, -1).astype(np.float32) / 255
        black_background = np.zeros_like(img, dtype=np.uint8)
        bgr_img = np.clip(
            img.astype(np.float32) * blurred_mask + black_background.astype(np.float32) * (1 - blurred_mask), a_min=0,
            a_max=255).astype(np.uint8)
        cv2.imwrite(f'alpha_image_with_{args.model_type}.png', bgr_img)
    else:
        raise Exception(f'{args.image_type} not found')
