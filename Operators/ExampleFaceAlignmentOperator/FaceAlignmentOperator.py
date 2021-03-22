from abc import ABC

import cv2
import numpy as np

from Operators.DummyAlgorithmWithModel import DummyAlgorithmWithModel
from Utils.GeometryUtils import center_pad_image_with_specific_base, \
    resize_with_long_side, force_convert_image_to_bgr
from Utils.InferenceHelpers.BaseInferenceHelper import TritonInferenceHelper


class FaceAlignmentOperator(DummyAlgorithmWithModel, ABC):
    name = 'FaceAlignment'
    __version__ = 'v1.0.20210322'

    def __init__(self, _inference_config, _is_test):
        super().__init__(_inference_config, _is_test)


class GeneralLandmark106p(FaceAlignmentOperator):
    """
    这个模型超级轻量级，速度快
    人脸轮廓效果会差点
    """
    name = '自然场景下的基于坐标点回归的106点landmark检测'
    __version__ = 'v1.0.20210322'

    def __init__(self, _inference_config, _is_test):
        super().__init__(_inference_config, _is_test)
        self.candidate_image_size = (192, 192)

    def get_inference_helper(self):
        if self.inference_config['name'] == 'triton':
            inference_helper = TritonInferenceHelper('Landmark106p',
                                                     self.inference_config['triton_url'],
                                                     self.inference_config['triton_port'],
                                                     'Landmark106p', 1)
            inference_helper.add_image_input('INPUT__0', (192, 192, 3), '检测用rgb的图像',
                                             ([0, 0, 0], [1, 1, 1]))
            inference_helper.add_output('OUTPUT__0', (1, 212), '回归的坐标')
            self.inference_helper = inference_helper
        else:
            raise NotImplementedError(
                f"{self.inference_config['name']} helper for face alignment 106p not implement")

    def execute(self, _image):
        to_return_result = {
            'points_count': 106,
            'x_locations': [0] * 106,
            'y_locations': [0] * 106,
        }
        resized_image = resize_with_long_side(_image, 192)
        resized_h, resized_w = resized_image.shape[:2]
        padded_image, (width_pad_ratio, height_pad_ratio) = center_pad_image_with_specific_base(
            resized_image,
            _width_base=192,
            _height_base=192,
            _output_pad_ratio=True
        )
        candidate_image = cv2.cvtColor(force_convert_image_to_bgr(padded_image), cv2.COLOR_BGR2RGB)
        candidate_h, candidate_w = candidate_image.shape[:2]
        if isinstance(self.inference_helper, TritonInferenceHelper):
            result = self.inference_helper.infer(_need_tensor_check=False, INPUT__0=candidate_image.astype(np.float32))
            coordinates = result['OUTPUT__0'].squeeze()
        else:
            raise NotImplementedError(
                f"{self.inference_helper.type_name} helper for 106p landmark not implement")
        remapped_coordinates = np.reshape(coordinates, (-1, 2))
        to_return_result['x_locations'] = \
            ((remapped_coordinates[:, 0] + 1) * (candidate_w // 2) - width_pad_ratio * candidate_w) / resized_w
        to_return_result['y_locations'] = \
            ((remapped_coordinates[:, 1] + 1) * (candidate_h // 2) - height_pad_ratio * candidate_h) / resized_h
        return to_return_result


if __name__ == '__main__':
    from argparse import ArgumentParser
    from Utils.AnnotationTools import annotate_circle_on_image
    from Operators.ExampleFaceDetectOperator.FaceDetectOperator import GeneralUltraLightFaceDetect
    from Utils.GeometryUtils import get_rotated_box_roi_from_image

    ag = ArgumentParser('Face Alignment Example')
    ag.add_argument('-i', '--image_path', dest='image_path', type=str, required=True, help='本地图像路径')
    ag.add_argument('-u', '--triton_url', dest='triton_url', type=str, required=True, help='triton url')
    ag.add_argument('-p', '--triton_port', dest='triton_port', type=int, default=8001, help='triton grpc 端口')
    args = ag.parse_args()
    # 假设图中只有一个人头
    img = cv2.imread(args.image_path)
    landmark106p_detect_handler = GeneralLandmark106p({
        'name': 'triton',
        'triton_url': args.triton_url,
        'triton_port': args.triton_port
    }, True)
    ultra_light_face_detect_handler = GeneralUltraLightFaceDetect({
        'name': 'triton',
        'triton_url': args.triton_url,
        'triton_port': args.triton_port
    }, True, 0.7, 0.5)
    landmark106p_result = landmark106p_detect_handler.execute(img)
    landmark106p_result_image = img.copy()
    landmark106p_all_points = [(x, y) for x, y in
                               zip(landmark106p_result['x_locations'],
                                   landmark106p_result['y_locations'])
                               ]
    annotate_circle_on_image(landmark106p_result_image, landmark106p_all_points, (0, 255, 0), 3, -1)
    cv2.imshow('landmark106p_result_image', landmark106p_result_image)
    face_detect_result = ultra_light_face_detect_handler.execute(img)
    face_bbox = face_detect_result['locations'][0]['box_info']
    cropped_face_region = get_rotated_box_roi_from_image(img, face_bbox, 1.25)
    landmark106p_with_bbox_result = landmark106p_detect_handler.execute(cropped_face_region)
    landmark106p_with_bbox_result_image = cropped_face_region.copy()
    landmark106p_with_bbox_result_all_points = [(x, y) for x, y in
                                                zip(landmark106p_with_bbox_result['x_locations'],
                                                    landmark106p_with_bbox_result['y_locations'])
                                                ]
    annotate_circle_on_image(landmark106p_with_bbox_result_image, landmark106p_with_bbox_result_all_points,
                             (255, 0, 255), 3, -1)
    cv2.imshow('landmark106p_with_bbox_result_image', landmark106p_with_bbox_result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
