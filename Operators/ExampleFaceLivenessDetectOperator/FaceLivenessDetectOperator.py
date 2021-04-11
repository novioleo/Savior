from abc import ABC

from scipy.special import softmax

from Operators.DummyAlgorithmWithModel import DummyAlgorithmWithModel
from Utils.InferenceHelpers import TritonInferenceHelper
import cv2
import numpy as np


class FaceLivenessDetect(DummyAlgorithmWithModel, ABC):
    name = '人脸活体检测'
    __version__ = 'v1.0.20210323'


class GeneralMiniFASNetV1SE(FaceLivenessDetect):
    name = '基于MiniFASNetV1SE的静默人脸活体检测'
    __version__ = 'v1.0.20210325'

    def get_inference_helper(self):
        if self.inference_config['name'] == 'triton':
            inference_helper = TritonInferenceHelper(
                'MiniFASNetV1SE',
                self.inference_config['triton_url'],
                self.inference_config['triton_port'],
                'MiniFASNetV1SE',
                1
            )
            inference_helper.add_image_input('INPUT__0', (80, 80, 3), '人脸图像',
                                             ([0, 0, 0], [1, 1, 1]))
            inference_helper.add_output('OUTPUT__0', (1, 3), '三个类别的分类情况，0，2均为非真实人脸')
            self.inference_helper = inference_helper
        else:
            raise NotImplementedError(
                f"{self.inference_config['name']} helper for MiniFASNetV1SE not implement")

    def execute(self, _image):
        """
        需要使用扩充4倍的人脸区域
        """
        to_return_result = {
            'classification_score_count': 3,
            'classification_scores': [0] * 3
        }
        resized_image = cv2.resize(_image, (80, 80))
        candidate_image = force_convert_image_to_bgr(resized_image)
        if isinstance(self.inference_helper, TritonInferenceHelper):
            result = self.inference_helper.infer(_need_tensor_check=False, INPUT__0=candidate_image.astype(np.float32))
            scores = result['OUTPUT__0'].squeeze()
        else:
            raise NotImplementedError(
                f"{self.inference_helper.type_name} helper for MiniFASNetV1SE not implement")
        to_return_result['classification_scores'] = softmax(scores).astype(np.float32).tolist()
        return to_return_result


class GeneralMiniFASNetV2(FaceLivenessDetect):
    """
    结果会相较V1更加准确
    """

    name = '基于MiniFASNetV2的静默人脸活体检测'
    __version__ = 'v1.0.20210325'

    def get_inference_helper(self):
        if self.inference_config['name'] == 'triton':
            inference_helper = TritonInferenceHelper(
                'MiniFASNetV2',
                self.inference_config['triton_url'],
                self.inference_config['triton_port'],
                'MiniFASNetV2',
                1
            )
            inference_helper.add_image_input('INPUT__0', (80, 80, 3), '人脸图像',
                                             ([0, 0, 0], [1, 1, 1]))
            inference_helper.add_output('OUTPUT__0', (1, 3), '三个类别的分类情况，0，2均为非真实人脸')
            self.inference_helper = inference_helper
        else:
            raise NotImplementedError(
                f"{self.inference_config['name']} helper for MiniFASNetV2 not implement")

    def execute(self, _image):
        """
        需要使用扩充2.7倍的人脸区域
        """
        to_return_result = {
            'classification_score_count': 3,
            'classification_scores': [0] * 3
        }
        resized_image = cv2.resize(_image, (80, 80))
        candidate_image = force_convert_image_to_bgr(resized_image)
        if isinstance(self.inference_helper, TritonInferenceHelper):
            result = self.inference_helper.infer(_need_tensor_check=False, INPUT__0=candidate_image.astype(np.float32))
            scores = result['OUTPUT__0'].squeeze()
        else:
            raise NotImplementedError(
                f"{self.inference_helper.type_name} helper for MiniFASNetV2 not implement")
        to_return_result['classification_scores'] = softmax(scores).astype(np.float32).tolist()
        return to_return_result


if __name__ == '__main__':
    from argparse import ArgumentParser
    from Operators.ExampleFaceDetectOperator import GeneralUltraLightFaceDetect
    from Utils.GeometryUtils import get_rotated_box_roi_from_image, force_convert_image_to_bgr

    ag = ArgumentParser('Face Liveness Detect Example')
    ag.add_argument('-i', '--image_path', dest='image_path', type=str, required=True, help='本地图像路径')
    ag.add_argument('-u', '--triton_url', dest='triton_url', type=str, required=True, help='triton url')
    ag.add_argument('-p', '--triton_port', dest='triton_port', type=int, default=8001, help='triton grpc 端口')
    args = ag.parse_args()
    # 假设图中只有一个人头
    img = cv2.imread(args.image_path)
    ultra_light_face_detect_handler = GeneralUltraLightFaceDetect({
        'name': 'triton',
        'triton_url': args.triton_url,
        'triton_port': args.triton_port
    }, True, 0.7, 0.5)
    face_bbox = ultra_light_face_detect_handler.execute(img)['locations'][0]['box_info']
    cropped_image_2_7 = get_rotated_box_roi_from_image(img, face_bbox, 2.7)
    cropped_image_4_0 = get_rotated_box_roi_from_image(img, face_bbox, 4.0)
    cv2.imshow('cropped_image_4_0', cropped_image_4_0)
    cv2.imshow('cropped_image_2_7', cropped_image_2_7)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    mini_fasnetv1se_handler = GeneralMiniFASNetV1SE({
        'name': 'triton',
        'triton_url': args.triton_url,
        'triton_port': args.triton_port
    }, True)
    mini_fasnetv2_handler = GeneralMiniFASNetV2({
        'name': 'triton',
        'triton_url': args.triton_url,
        'triton_port': args.triton_port
    }, True)
    mini_fasnetv1se_result = mini_fasnetv1se_handler.execute(cropped_image_4_0)['classification_scores']
    mini_fasnetv2_result = mini_fasnetv1se_handler.execute(cropped_image_2_7)['classification_scores']
    combine_result = [sum(m_scores) for m_scores in zip(mini_fasnetv2_result, mini_fasnetv1se_result)]
    print(mini_fasnetv1se_result)
    print(mini_fasnetv2_result)
    if np.argmax(combine_result) == 1:
        print('real face')
    else:
        print('fake face')
